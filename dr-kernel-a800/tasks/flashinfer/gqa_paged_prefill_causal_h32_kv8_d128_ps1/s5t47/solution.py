import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_triplet_kernel(
    q_ptr,            # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,     # *float32, [num_pages, 8, 128] (squeezed from [num_pages, 1, 8, 128])
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, 32, 128] (will cast to bfloat16 on host)
    lse_ptr,          # *float32, [total_q, 32]
    sm_scale: tl.float32,  # scaling factor (e.g., 1/sqrt(128))
    GQA_RATIO: tl.constexpr,       # 4
    MAX_Q: tl.constexpr,           # e.g., 4096
    MAX_KV: tl.constexpr,          # e.g., 256
    HEAD_DIM: tl.constexpr,        # 128
):
    # Program ids
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Bounds for qo and kv
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate maximum index for keys seen by this query
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices)

    # GQA mapping: query head h uses KV head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Precompute global q index
    global_q_idx = qo_start + q_idx

    # Load q_sub vector for this (q_idx, h)
    q_off = global_q_idx * (32 * 128) + h * 128
    q_vec = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))  # [128]

    # First pass: compute lse = logsumexp of scaled dot products over i in [0, max_kv_idx)
    m = -float("inf")
    s = 0.0
    for i in tl.static_range(0, MAX_Q):  # We only need up to max_kv_idx; cap for compilation
        cond_i = i < max_kv_idx
        idx_i = kv_start + i  # Triton scalar int32
        # Row offsets for k and v: idx_i * (8 * 128) + kv_head * 128
        k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
        v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128
        # Masked loads
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Dot product q_sub @ k_row (scalar), masked by cond_i
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            qj = q_vec[j]
            kj = k_vec[j]
            dot += qj * kj
        dot = tl.where(cond_i, dot, 0.0)
        logits_scaled = dot * sm_scale
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(cond_i, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    lse_val = tl.log(s) + m  # logsumexp in natural log

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector for this (b, q_idx, h)
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    total_sum = 0.0
    for i in tl.static_range(0, MAX_Q):
        cond_i = i < max_kv_idx
        idx_i = kv_start + i
        k_row_ptr = k_ptr + idx_i * (8 * 128) + kv_head * 128
        v_row_ptr = v_ptr + idx_i * (8 * 128) + kv_head * 128
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale
        prob = tl.exp(logits_scaled - lse_val) * tl.where(cond_i, 1.0, 0.0)
        total_sum += tl.where(cond_i, prob, 0.0)
        out_vec += tl.where(cond_i, prob * v_vec, 0.0)

    # Store output vector
    out_off = global_q_idx * (32 * 128) + h * 128
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale: float):
        super().__init__()
        self.sm_scale = float(sm_scale)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale: float):
        # Cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        # Squeeze the (1,) dimension to match original: [num_pages, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        # Output and lse buffers (float32 for compute)
        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        num_pages, num_kv_heads, _, _ = k_cache.shape  # num_kv_heads should be 8
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert qo_indptr.shape[0] == kv_indptr.shape[0], "len_indptr must match for qo and kv"
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        out_f32 = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse_f32 = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (len_indptr, total_q, 32)
        _compute_triplet_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            out_f32, lse_f32,
            sm_scale=self.sm_scale,
            GQA_RATIO=4,
            MAX_Q=4096,
            MAX_KV=256,
            HEAD_DIM=128,
            num_warps=4,
            num_stages=2,
        )

        # Return output in bfloat16 (as in original) and lse in float32
        output_bf16 = out_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
