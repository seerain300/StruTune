import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_per_triplet_kernel(
    q_ptr,            # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr,            # *float32, [num_pages, num_kv_heads, head_dim] (we'll squeeze externally)
    v_ptr,            # *float32, [num_pages, num_kv_heads, head_dim]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *float32, [total_q, num_qo_heads]
    sm_scale: tl.float32,
    HEAD_DIM: tl.constexpr,       # e.g., 128
    NUM_QO_HEADS: tl.constexpr,   # e.g., 32
    NUM_KV_HEADS: tl.constexpr,   # e.g., 8
    MAX_KV: tl.constexpr,         # e.g., 256
    GQA_RATIO: tl.constexpr,      # 4
):
    # Each program handles one (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load qo_indptr[b] and qo_indptr[b+1] to get sequence range for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start

    # Candidate max for causal range: candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens)
    candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens)
    # Clamp candidate_max to [0, num_kv_indices_in_b]
    candidate_max = tl.maximum(0, tl.minimum(candidate_max, num_kv_indices_in_b))

    # Precompute small offsets
    q_off = q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    # Select corresponding KV head for GQA: kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # First pass: compute logsumexp over valid i
    m = -float("inf")  # running max
    s = 0.0            # running sum of exp relative to m
    for i in tl.static_range(0, MAX_KV):
        # Validity: i < candidate_max, i < num_kv_indices_in_b, and kv_start + i < kv_end
        valid = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)
        # idx_i = kv_indices[kv_start + i]
        idx_i = tl.load(kv_indices_ptr + (kv_start + i), mask=valid, other=0)

        # Build pointer to k_row and v_row for this idx_i and kv_head
        # k_ptr layout is [num_pages, NUM_KV_HEADS, HEAD_DIM], flattened in (NUM_KV_HEADS * HEAD_DIM) per row
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        # Compute dot product q_sub[h] @ k_vec
        q_vec = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale

        # Accumulate logsumexp
        m_new = tl.maximum(m, logits_scaled)
        # exp contribution: if valid, add exp(logits_scaled - m_new), else 0
        s = s * tl.exp(m - m_new) + tl.where(valid, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    lse_val = tl.log(s) + m  # natural logsumexp

    # Second pass: compute output vector and store
    out_off = q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    total_sum = 0.0
    for i in tl.static_range(0, MAX_KV):
        valid = (i < candidate_max) & (i < num_kv_indices_in_b) & (kv_start + i < kv_end)
        idx_i = tl.load(kv_indices_ptr + (kv_start + i), mask=valid, other=0)

        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        q_vec = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale

        # attention prob
        prob = tl.where(valid, tl.exp(logits_scaled - lse_val), 0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)
        out_vec += prob * v_vec

    # Store output vector
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])

    # Store lse for this (b, q_idx, h)
    lse_off = (q_idx * NUM_QO_HEADS + h)
    tl.store(lse_ptr + lse_off, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for computation
        q_f32 = q.to(torch.float32)
        k_cache_squeezed = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
        v_cache_squeezed = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

        total_q, num_qo_heads, head_dim = q_f32.shape
        assert num_qo_heads == 32
        assert head_dim == 128

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

        # Return output (float32) and lse (float32)
        return out, lse


def run(*args):
    return ModelNew()(*args)
