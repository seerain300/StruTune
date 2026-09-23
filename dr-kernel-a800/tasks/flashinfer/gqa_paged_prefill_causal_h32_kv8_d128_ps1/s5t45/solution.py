import math
import torch

import triton
import triton.language as tl


@triton.jit
def _attention_per_triplet_kernel(
    q_ptr,           # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,    # *float32, [num_pages, 8, 128] (k/v caches squeezed from original)
    qo_indptr_ptr,   # *int32, [len_indptr]
    kv_indptr_ptr,   # *int32, [len_indptr]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    out_ptr,         # *float32, [total_q, 32, 128]
    lse_ptr,         # *float32, [total_q, 32]
    GQA_RATIO: tl.constexpr,   # 32 // 8 = 4
    SM_SCALE: tl.constexpr,    # 1.0 / sqrt(128)
    MAX_Q: tl.constexpr,       # e.g., 4096
    MAX_KV: tl.constexpr,      # e.g., 256
    HEAD_DIM: tl.constexpr,    # 128
):
    # Each program handles one triplet (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load sequence boundaries for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # GQA mapping: query head h uses KV head kv_head
    kv_head = h // GQA_RATIO

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate maximum number of keys needed for causal + density
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices)

    # Global query index
    global_q_idx = qo_start + q_idx

    # Prepare vector for q[h]
    # q_ptr offset for global_q_idx and head h: global_q_idx * (32*128) + h * 128
    q_vec_ptr = q_ptr + global_q_idx * (32 * 128) + h * 128
    q_vec = tl.load(q_vec_ptr + tl.arange(0, HEAD_DIM))

    # Running max and sum for logsumexp
    m = -1e20  # scalar
    s = 0.0    # scalar

    # First pass: compute logsumexp of logits over valid i
    for i in tl.static_range(0, MAX_Q):
        # Skip out-of-range q_idx
        valid_q = i < num_q_tokens
        # Only compute if valid_q and i == q_idx
        if valid_q and (i == q_idx):
            for i2 in tl.static_range(0, MAX_KV):
                # Valid i2 if i2 < max_kv_idx
                cond_i2 = i2 < max_kv_idx
                idx_i2 = kv_start + i2
                # Masked loads for k_row and v_row
                k_row_ptr = k_ptr + idx_i2 * (8 * 128) + kv_head * 128
                k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i2, other=0.0)

                v_row_ptr = v_ptr + idx_i2 * (8 * 128) + kv_head * 128
                v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i2, other=0.0)

                # Dot product between q_vec[h] and k_row (scalar)
                dot = 0.0
                for j in tl.static_range(0, HEAD_DIM):
                    dot += q_vec[j] * k_vec[j]
                logits_scaled = dot * SM_SCALE

                # Update running max and sum for logsumexp
                m_new = tl.maximum(m, logits_scaled)
                s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(cond_i2, tl.exp(logits_scaled - m_new), 0.0)
                m = m_new

    # Compute logsumexp
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    total_sum = 0.0
    for i in tl.static_range(0, MAX_Q):
        valid_q = i < num_q_tokens
        if valid_q and (i == q_idx):
            for i2 in tl.static_range(0, MAX_KV):
                cond_i2 = i2 < max_kv_idx
                idx_i2 = kv_start + i2

                k_row_ptr = k_ptr + idx_i2 * (8 * 128) + kv_head * 128
                k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i2, other=0.0)

                v_row_ptr = v_ptr + idx_i2 * (8 * 128) + kv_head * 128
                v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i2, other=0.0)

                # Recompute dot
                dot = 0.0
                for j in tl.static_range(0, HEAD_DIM):
                    dot += q_vec[j] * k_vec[j]
                logits_scaled = dot * SM_SCALE
                prob = tl.exp(logits_scaled - lse_val) * tl.where(cond_i2, 1.0, 0.0)
                out_vec += prob * v_vec

    # Store output vector for this (b, q_idx, h)
    out_off = global_q_idx * (32 * 128) + h * 128
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k_cache: [num_pages, 1, 8, 128], bfloat16
        v_cache: [num_pages, 1, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float
        """
        device = q.device
        # Ensure dtype float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        # Squeeze the (1,) dimension for k_cache/v_cache
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        num_pages = k_cache_flat.shape[0]
        num_kv_heads = k_cache_flat.shape[1]
        _k_dim = k_cache_flat.shape[2]
        _h_dim = k_cache_flat.shape[3]
        # Sanity checks
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert _k_dim == 128 and _h_dim == 128, "k_cache/v_cache last two dims must be [8, 128] after squeeze"

        len_indptr = qo_indptr.shape[0]
        assert kv_indptr.shape[0] == len_indptr, "qo_indptr and kv_indptr must have same length"

        # Allocate outputs
        output_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid = (len_indptr, total_q, num_qo_heads)
        grid = (len_indptr, total_q, num_qo_heads)

        _attention_per_triplet_kernel[grid](
            q_f32,
            k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output_f32, lse_f32,
            GQA_RATIO=4,
            SM_SCALE=float(sm_scale),
            MAX_Q=4096,
            MAX_KV=256,
            HEAD_DIM=128,
        )

        # Return output as bfloat16 and lse as float32 to match original
        output_bf16 = output_f32.to(torch.bfloat16)
        return output_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
