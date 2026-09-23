import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_per_batch_kernel(
    q_ptr,            # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,     # *float32, [num_pages, num_kv_heads, head_dim] (squeezed from original)
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *float32, [total_q, num_qo_heads]
    GQA_RATIO: tl.constexpr,   # 4 (32 query heads -> 8 KV heads)
    HEAD_DIM: tl.constexpr,    # 128
    MAX_KV: tl.constexpr,      # e.g., 256
    sm_scale: tl.float32,      # scaling factor
):
    # Program ids: each program handles one (q_idx, h, b)
    q_idx = tl.program_id(0)  # total_q
    h = tl.program_id(1)      # num_qo_heads
    b = tl.program_id(2)      # len_indptr - 1

    # Load qo_indptr[b] and qo_indptr[b+1] to get sequence range for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start  # scalar
    num_kv_indices = kv_end - kv_start  # scalar

    # Precompute candidate max for each query q_idx:
    # candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    # Note: Triton needs all math to be Triton tensors here
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)

    # GQA mapping: query head h uses KV head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Load query vector q_sub = q[q_idx, h, :]
    # q_ptr layout: (total_q, num_qo_heads, HEAD_DIM)
    q_off = q_idx * (32 * 128) + h * 128  # qo_indptr/qv dimensions are int, but we index by q_idx and h
    q_vec = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))

    # Running max and sum for logsumexp over valid i
    m = -float("inf")
    s = 0.0

    # First pass: compute logsumexp over valid keys
    for i in tl.static_range(0, MAX_KV):
        # Validity conditions: i < candidate_max, i < num_kv_indices, kv_start + i < kv_end
        valid = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        # Compute offsets for k_ptr and v_ptr: [num_pages, num_kv_heads, HEAD_DIM]
        # idx_i selects the row, kv_head selects the KV head, j selects the head_dim
        # Pointer arithmetic in Triton: use Triton scalars
        k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        # Dot product: q_sub[h] @ k_row -> scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]

        logits_scaled = dot * sm_scale
        # Update running max and sum
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(valid, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    # Compute lse
    lse_val = tl.log(s) + m  # natural logsumexp

    # Store lse for this (q_idx, h, b)
    global_q_idx = qo_start + q_idx
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector out[q_idx, h, :] = sum_i prob_i * v_row
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    total_sum = 0.0

    for i in tl.static_range(0, MAX_KV):
        valid = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]

        logits_scaled = dot * sm_scale
        prob = tl.exp(logits_scaled - lse_val) * tl.where(valid, 1.0, 0.0)
        out_vec += prob * v_vec

    # Store output vector for this (q_idx, h, b)
    out_off = (q_idx * 32 + h) * 128
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])

    # We processed this batch; no need to write back lse again per batch.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, 32, 128], dtype bfloat16
        k_cache: [num_pages, 1, 8, 128], dtype bfloat16
        v_cache: [num_pages, 1, 8, 128], dtype bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar
        Returns:
        - output: [total_q, 32, 128], bfloat16
        - lse: [total_q, 32], float32
        """
        # Ensure inputs are on the same device and dtype conversions
        device = q.device
        q_f32 = q.to(torch.float32)
        # Squeeze the (1,) dimension
        k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        # Allocate output and lse
        total_q = q_f32.shape[0]
        num_qo_heads = 32
        head_dim = 128
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (q_idx, h, b). b ranges from 0 to len_indptr - 2.
        # Note: We must have at least one batch (len_indptr >= 2). The original code asserts len_indptr>1.
        len_indptr = qo_indptr.shape[0]
        grid = (total_q, num_qo_heads, len_indptr - 1)

        _attention_per_batch_kernel[grid](
            q_f32,
            k_cache_flat,
            v_cache_flat,
            qo_indptr,
            kv_indptr,
            kv_indices,
            output,
            lse,
            GQA_RATIO=4,
            HEAD_DIM=128,
            MAX_KV=256,
            sm_scale=sm_scale,
        )

        # Cast output back to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
