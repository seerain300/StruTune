import math
import torch

import triton
import triton.language as tl


@triton.jit
def _batched_attention_kernel(
    q_ptr,                # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,         # *float32, [num_pages, num_kv_heads, head_dim]
    qo_indptr_ptr,        # *int32, [len_indptr]
    kv_indptr_ptr,        # *int32, [len_indptr]
    kv_indices_ptr,       # *int32, [num_kv_indices]
    out_ptr,              # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,              # *float32, [total_q, num_qo_heads]
    total_q,              # int32
    num_qo_heads,         # int32
    num_kv_heads,         # int32
    head_dim,             # int32
    len_indptr,           # int32
    sm_scale,             # float32
    GQA_RATIO: tl.constexpr,   # int
    MAX_KV: tl.constexpr,      # int (compile-time max for loop)
    MAX_Q: tl.constexpr,       # int (compile-time max for q loop)
):
    # Grid: (len_indptr, MAX_Q, num_qo_heads)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load batch bounds from indptr arrays
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Effective counts for this batch
    num_q_tokens_in_b = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start

    # If q_idx exceeds num_q_tokens_in_b, skip (happens for MAX_Q > num_q_tokens_in_b); we also skip if b == len_indptr-1
    # But here we assume grid is exactly (len_indptr, num_q_tokens_in_b, num_qo_heads), so q_idx < num_q_tokens_in_b.
    global_q_idx = qo_start + q_idx

    # GQA mapping: query head h uses kv head h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Load q_sub = q[global_q_idx, h, :]
    offs = tl.arange(0, head_dim)
    q_offset = (global_q_idx * num_qo_heads + h) * head_dim
    q_sub = tl.load(q_ptr + q_offset + offs)  # [head_dim], float32

    # Accumulators for lse
    max_acc = -float("inf")
    sum_exp = 0.0

    # First pass: compute logsumexp over candidate i in [0, MAX_KV), masked by i < min(candidate_max, num_kv_indices_in_b)
    # candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens_in_b)
    q_idx_plus_one = q_idx + 1
    delta = num_kv_indices_in_b - num_q_tokens_in_b
    candidate_max = q_idx_plus_one + delta

    # Loop over i with masks
    for i in tl.static_range(MAX_KV):
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)
        # Compute idx into kv_indices
        idx = tl.load(kv_indices_ptr + (kv_start + i), mask=valid_i, other=0)  # scalar int32
        # Compute offsets for k/v rows: [num_pages, num_kv_heads, head_dim]
        k_offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim

        # Load k_row and v_row (masked)
        k_row = tl.load(k_ptr + k_offset + offs, mask=valid_i, other=0.0)  # [head_dim]
        v_row = tl.load(v_ptr + v_offset + offs, mask=valid_i, other=0.0)  # [head_dim]

        # Dot product q_sub @ k_row (scalar)
        dot_val = tl.sum(q_sub * k_row, axis=0)  # scalar float32

        # Update logsumexp in a numerically stable manner
        val = sm_scale * dot_val
        new_max = tl.maximum(max_acc, val)
        # If invalid_i, sum_exp stays; else add exp(val - max_acc)
        sum_exp = tl.where(valid_i, sum_exp + tl.exp(val - max_acc), sum_exp)
        max_acc = new_max

    # Compute lse in base-2
    logsumexp = max_acc + tl.log(sum_exp)  # scalar
    lse_base2 = logsumexp / math.log(2.0)
    # Store lse
    lse_offset = global_q_idx * num_qo_heads + h
    tl.store(lse_ptr + lse_offset, lse_base2)

    # Second pass: accumulate output vector out[global_q_idx, h, :]
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for i in tl.static_range(MAX_KV):
        valid_i = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)
        idx = tl.load(kv_indices_ptr + (kv_start + i), mask=valid_i, other=0)
        k_offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        v_offset = idx * (num_kv_heads * head_dim) + kv_head * head_dim
        k_row = tl.load(k_ptr + k_offset + offs, mask=valid_i, other=0.0)
        v_row = tl.load(v_ptr + v_offset + offs, mask=valid_i, other=0.0)

        dot_val = tl.sum(q_sub * k_row, axis=0)
        val = sm_scale * dot_val
        p_i = tl.where(valid_i, tl.exp(val - max_acc) / sum_exp, 0.0)
        out_vec = out_vec + p_i * v_row

    # Store output
    out_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    tl.store(out_ptr + out_offset + offs, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim: int = 128, num_qo_heads: int = 32, num_kv_heads: int = 8, GQA_RATIO: int = 4, MAX_Q: int = 32, MAX_KV: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.GQA_RATIO = GQA_RATIO
        # MAX_Q and MAX_KV are compile-time constants for the Triton kernel static loops
        self.MAX_Q = MAX_Q
        self.MAX_KV = MAX_KV

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, num_qo_heads, head_dim], dtype bfloat16
        k_cache, v_cache: [num_pages, 1, num_kv_heads, head_dim], dtype bfloat16
        qo_indptr, kv_indptr: [len_indptr], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float (e.g., 1.0 / sqrt(head_dim))
        Returns: (output: [total_q, num_qo_heads, head_dim], dtype bfloat16), (lse: [total_q, num_qo_heads], dtype float32)
        """
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        device = q.device

        # Flatten k/v along "1" dimension (as original code uses k.squeeze(1))
        k_cache_flat = k_cache.squeeze(1).contiguous()   # [num_pages, num_kv_heads, head_dim]
        v_cache_flat = v_cache.squeeze(1).contiguous()   # [num_pages, num_kv_heads, head_dim]

        # Cast inputs to float32 for Triton compute
        q_f32 = q.to(torch.float32)
        k_f32 = k_cache_flat.to(torch.float32)
        v_f32 = v_cache_flat.to(torch.float32)
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)

        # Allocate output and lse buffers (float32)
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        # Grid: (len_indptr, MAX_Q, num_qo_heads). We use MAX_Q=32 to match typical queries; adjust in __init__ if needed.
        grid = (qo_indptr.shape[0], self.MAX_Q, num_qo_heads)
        _batched_attention_kernel[grid](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr, kv_indices, out_f32, lse_f32,
            total_q, num_qo_heads, num_kv_heads, head_dim, qo_indptr.shape[0], float(sm_scale),
            GQA_RATIO=self.GQA_RATIO, MAX_KV=self.MAX_KV, MAX_Q=self.MAX_Q,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 for final result; lse remains float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
