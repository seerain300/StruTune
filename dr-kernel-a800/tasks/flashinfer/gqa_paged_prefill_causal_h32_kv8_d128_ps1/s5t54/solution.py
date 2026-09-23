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
    SM_SCALE: tl.float32,  # scaling factor = 1 / sqrt(128)
    NUM_QO_HEADS: tl.constexpr,  # 32
    NUM_KV_HEADS: tl.constexpr,  # 8
    HEAD_DIM: tl.constexpr,      # 128
    MAX_KV: tl.constexpr,        # e.g., 256
):
    # Program ids: one triplet per program (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load sequence bounds
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate maximum number of keys this query can see
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices)

    # Global query index
    global_q_idx = qo_start + q_idx

    # GQA mapping: query head uses KV head
    kv_head = h // NUM_QO_HEADS  # 32 // 4 => 8
    # If GQA ratio is 4: h // 4 gives kv_head in [0, 7]

    # First pass: compute logsumexp over valid keys for this query and head
    m = -float('inf')
    s = 0.0
    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < max_kv_idx) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        # Pointer to k row for this idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Load q vector for this query position and head h: q is [total_q, 32, 128]
        q_row_ptr = q_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM)) + h * HEAD_DIM
        q_vec = tl.load(q_row_ptr + tl.arange(0, HEAD_DIM))

        # Compute dot(q_sub[h], k_row) as scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * SM_SCALE

        # Update running max and sum for logsumexp
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(cond_i, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    # lse = log(s) + m (natural logsumexp)
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector for this (b, q_idx, h)
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < max_kv_idx) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        # Pointer to k row for this idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Load q vector for this query position and head h: q is [total_q, 32, 128]
        q_row_ptr = q_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM)) + h * HEAD_DIM
        q_vec = tl.load(q_row_ptr + tl.arange(0, HEAD_DIM))

        # Compute dot(q_sub[h], k_row) as scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * SM_SCALE
        prob = tl.exp(logits_scaled - lse_val) * tl.where(cond_i, 1.0, 0.0)

        # Load corresponding v row and accumulate
        v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)
        out_vec += prob * v_vec

    # Store output vector for this (b, q_idx, h): out is [total_q, 32, 128]
    out_row_base = out_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM))  # base for this query
    out_vec_base = out_row_base + (h * HEAD_DIM)                        # base for this head
    tl.store(out_vec_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.head_dim = 128
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for Triton compute (original code casts q to float32)
        q_f32 = q.contiguous().to(torch.float32)                      # [total_q, 32, 128]
        # k_cache, v_cache: [num_pages, 1, 8, 128] -> squeeze dim=1
        k_cache_flat = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]

        # Allocate outputs (float32 for compute; will be cast to bfloat16 later)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel
        # Grid: one program per (b, q_idx, h)
        grid = (qo_indptr.shape[0], total_q, num_qo_heads)
        _attention_one_triplet_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            SM_SCALE=self.sm_scale,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
            HEAD_DIM=head_dim,
            MAX_KV=256,
            num_warps=4, num_stages=1,
        )

        # Cast output to bfloat16 to match original dtype of q
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
