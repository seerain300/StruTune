import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_per_batch_kernel(
    q_ptr,            # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,     # *float32, [num_pages, 8, 128] (squeezed)
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, 32, 128]
    lse_ptr,          # *float32, [total_q, 32]
    NUM_PAGES: tl.constexpr,   # e.g., 51
    NUM_QO_HEADS: tl.constexpr, # 32
    NUM_KV_HEADS: tl.constexpr, # 8
    HEAD_DIM: tl.constexpr,     # 128
    GQA_RATIO: tl.constexpr,    # 4
    MAX_KV: tl.constexpr,       # e.g., 256
    sm_scale: tl.float32,       # scaling factor
):
    # Program ids: (q_idx, h, b)
    q_idx = tl.program_id(0)  # in [0, total_q)
    h = tl.program_id(1)      # in [0, 32)
    b = tl.program_id(2)      # in [0, len_indptr - 2] (host launches up to len_indptr - 1)

    # Load start/end for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Precompute candidate max for each query q_idx: candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    candidate_max = tl.max(0, candidate_max)
    max_kv_idx = tl.min(MAX_KV, candidate_max)

    # GQA mapping: query head h uses KV head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO

    # Load q_sub vector for this (q_idx, h)
    q_row_ptr = q_ptr + q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_vec = tl.load(q_row_ptr + tl.arange(0, HEAD_DIM))

    # Running max and sum for logsumexp across i
    m = -float("inf")
    s = 0.0

    # First pass: compute logsumexp over all valid i
    for i in tl.static_range(0, MAX_KV):
        # Validity conditions
        cond_i = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        # Load k_row and v_row for this idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_PAGES * NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_PAGES * NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Compute dot = q_vec @ k_vec.T -> scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale

        # Update running max and sum for logsumexp
        m_new = tl.maximum(m, logits_scaled)
        # s_new = s * exp(m - m_new) + exp(m_new - m) * exp(logits_scaled - m_new) where cond_i
        s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(cond_i, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    # Compute lse as logsumexp
    lse_val = tl.log(s) + m  # natural logsumexp
    tl.store(lse_ptr + q_idx * NUM_QO_HEADS + h, lse_val)

    # Second pass: compute output vector for this (q_idx, h) across all i
    out_row = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        k_row_ptr = k_ptr + idx_i * (NUM_PAGES * NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_PAGES * NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale
        prob = tl.exp(logits_scaled - lse_val) * tl.where(cond_i, 1.0, 0.0)
        out_row += prob * v_vec

    # Store output row for this (q_idx, h)
    out_row_ptr = out_ptr + q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_row_ptr + j, out_row[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes and assertions
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "k_cache/v_cache must have a squeezed (1,) dimension"
        assert k_cache.shape[3] == 128 and v_cache.shape[3] == 128, "k_cache/v_cache head_dim must be 128"
        assert qo_indptr.shape[0] == len_indptr and kv_indptr.shape[0] == len_indptr, "indptr shapes must match len_indptr"
        assert q.device.type == "cuda" and k_cache.device.type == "cuda" and v_cache.device.type == "cuda", "Tensors must be on CUDA device"

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        # Squeeze the (1,) dimension to match [num_pages, num_kv_heads, head_dim]
        k_cache_flat = k_cache.squeeze(1).contiguous()
        v_cache_flat = v_cache.squeeze(1).contiguous()

        # Allocate outputs (float32 for kernel, then cast output to bfloat16)
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel per batch: grid over (q_idx, h, b)
        grid = (total_q, num_qo_heads, len_indptr - 1)
        _attention_per_batch_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            out, lse,
            NUM_PAGES=num_pages,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=4,
            MAX_KV=256,
            sm_scale=sm_scale,
        )

        # Return output in bfloat16 (original dtype) and lse in float32
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
