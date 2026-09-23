import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_per_triplet_kernel(
    q_ptr,           # *float32, [total_q, num_qo_heads, head_dim] (row-major)
    k_ptr, v_ptr,    # *float32, [num_pages, num_kv_heads, head_dim] (squeezed from original)
    qo_indptr_ptr,   # *int32, [len_indptr]
    kv_indptr_ptr,   # *int32, [len_indptr]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    out_ptr,         # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,         # *float32, [total_q, num_qo_heads]
    sm_scale: tl.float32,
    HEAD_DIM: tl.constexpr,   # 128
    NUM_QO_HEADS: tl.constexpr,  # 32
    NUM_KV_HEADS: tl.constexpr,  # 8
    MAX_KV: tl.constexpr,        # e.g., 256
    GQA_RATIO: tl.constexpr,     # 4
):
    # Program ids: one program per (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load qo indptrs for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate max for this q_idx: candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)

    # GQA mapping: query head h uses KV head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Global q index for this query
    global_q_idx = qo_start + q_idx

    # Load q_sub vector for this head: q[qo_start + q_idx, h]
    q_row_ptr = q_ptr + (qo_start + q_idx) * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_vec = tl.load(q_row_ptr + tl.arange(0, HEAD_DIM))

    # Running max and sum for logsumexp across all valid i
    m = -float("inf")
    s = 0.0

    # First pass: compute lse
    for i in tl.static_range(0, MAX_KV):
        # Validity: i < candidate_max, i < num_kv_indices, kv_start + i < kv_end
        valid = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        if valid:
            # idx_i = kv_indices[kv_start + i]
            idx_i = tl.load(kv_indices_ptr + (kv_start + i))
            # Pointer arithmetic in Triton: use Triton scalars
            k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM))
            # Dot product: q_sub[h] @ k_vec
            dot = 0.0
            for j in tl.static_range(0, HEAD_DIM):
                dot += q_vec[j] * k_vec[j]
            logits_scaled = dot * sm_scale
            m_new = tl.maximum(m, logits_scaled)
            # Update sum with rescaling
            s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.exp(logits_scaled - m_new)
            m = m_new

    # Compute lse (natural logsumexp)
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * NUM_QO_HEADS + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector for this head
    out_row_ptr = out_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        valid = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        if valid:
            idx_i = tl.load(kv_indices_ptr + (kv_start + i))
            k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM))
            dot = 0.0
            for j in tl.static_range(0, HEAD_DIM):
                dot += q_vec[j] * k_vec[j]
            logits_scaled = dot * sm_scale
            prob = tl.exp(logits_scaled - lse_val)
            v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM))
            out_vec += prob * v_vec

    # Store output vector for this (b, q_idx, h)
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_row_ptr + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtypes
        total_q, num_qo_heads, head_dim = q.shape
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # Squeeze the (1,) dimension for k_cache/v_cache
        k_cache_squeezed = k_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]
        v_cache_squeezed = v_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]

        # Allocate outputs
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        len_indptr = qo_indptr.shape[0]
        grid = (len_indptr - 1, total_q, num_qo_heads)
        _attention_per_triplet_kernel[grid](
            q_f32, k_cache_squeezed, v_cache_squeezed,
            qo_indptr, kv_indptr, kv_indices,
            out, lse,
            sm_scale,
            HEAD_DIM=128, NUM_QO_HEADS=32, NUM_KV_HEADS=8,
            MAX_KV=256, GQA_RATIO=4,
            num_warps=4, num_stages=2
        )

        # Return output and lse
        return out, lse


def run(*args):
    return ModelNew()(*args)
