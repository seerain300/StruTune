import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_one_triplet_kernel(
    q_ptr,            # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,     # *float32, [num_pages, 8, 128] (k_cache squeezed from [num_pages, 1, 8, 128])
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, 32, 128]
    lse_ptr,          # *float32, [total_q, 32]
    SM_SCALE: tl.float32,
    HEAD_DIM: tl.constexpr,          # 128
    NUM_KV_HEADS: tl.constexpr,      # 8
    GQA_RATIO: tl.constexpr,         # 4
    MAX_KV: tl.constexpr,            # e.g., 256
):
    # Program ids: (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load sequence bounds
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries/keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate maximum number of keys to consider for this query
    delta = num_kv_indices - num_q_tokens
    candidate_max = q_idx + 1 + delta
    # Clamp candidate_max to [0, num_kv_indices]
    candidate_max = candidate_max if candidate_max >= 0 else 0
    candidate_max = candidate_max if candidate_max <= num_kv_indices else num_kv_indices
    max_kv_idx = candidate_max  # scalar int32

    # Global query index
    global_q_idx = qo_start + q_idx

    # GQA mapping: query head h uses KV head kv_head = h // GQA_RATIO
    kv_head = h // GQA_RATIO  # h in [0, 31], GQA_RATIO=4 -> kv_head in [0, 7]

    # Load q_sub[h] as vector
    q_row_base = q_ptr + (global_q_idx * (32 * HEAD_DIM))  # q index base
    q_vec_base = q_row_base + (h * HEAD_DIM)              # head base
    q_vec = tl.load(q_vec_base + tl.arange(0, HEAD_DIM))  # [128], float32

    # First pass: compute logsumexp over valid i
    m = tl.full((), -float("inf"), tl.float32)  # running max for logsumexp
    s = tl.zeros((), dtype=tl.float32)          # running sum in exp space

    for i in tl.static_range(0, MAX_KV):
        # Validity
        cond_i = (i < max_kv_idx) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i  # scalar int32

        # Load k_row and v_row for idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Compute dot(q_sub, k_row) as scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * SM_SCALE

        # Update running max and sum
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(logits_scaled - m_new) * tl.where(cond_i, 1.0, 0.0)
        m = m_new

    # lse = log(s) + m (natural logsumexp)
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < max_kv_idx) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * SM_SCALE
        prob = tl.exp(logits_scaled - lse_val) * tl.where(cond_i, 1.0, 0.0)
        out_vec += prob * v_vec

    # Store output vector for this (b, q_idx, h)
    out_row_base = out_ptr + (global_q_idx * (32 * HEAD_DIM))  # output index base
    out_vec_base = out_row_base + (h * HEAD_DIM)              # head base
    tl.store(out_vec_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128


def run(*args):
    return ModelNew()(*args)
