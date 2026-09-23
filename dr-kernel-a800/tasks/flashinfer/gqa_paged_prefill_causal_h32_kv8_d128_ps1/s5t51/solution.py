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
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    MAX_KV: tl.constexpr,         # e.g., 256
):
    # Program ids: each program handles one (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load qo_indptr[b] and qo_indptr[b+1] to get sequence range for this batch
    qo_start = tl.load(qo_indptr_ptr + b)  # int32
    qo_end = tl.load(qo_indptr_ptr + b + 1)  # int32

    # Load kv_indptr[b] and kv_indptr[b+1] to get key ranges for this batch
    kv_start = tl.load(kv_indptr_ptr + b)  # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32

    # Number of queries and keys in this batch
    num_q_tokens = qo_end - qo_start
    num_kv_indices = kv_end - kv_start

    # Candidate max for this query: q_idx + 1 + (num_kv_indices - num_q_tokens)
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)

    # global_q_idx is just q_idx (since sequences are contiguous within b)
    global_q_idx = q_idx

    # GQA mapping: query head h uses KV head kv_head = h // 4
    kv_head = h // 4

    # Load q_sub[h] (vector of length HEAD_DIM) as float32
    q_row_base = q_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM))  # base for this query in q
    q_vec_base = q_row_base + (h * HEAD_DIM)                        # base for this head
    q_vec = tl.load(q_vec_base + tl.arange(0, HEAD_DIM))

    # Compute max_kv_idx
    max_kv_idx = tl.maximum(0, candidate_max)
    max_kv_idx = tl.minimum(max_kv_idx, num_kv_indices)  # ensure within bounds

    # First pass: compute logsumexp of scaled logits across all candidate i
    m = -float("inf")
    s = 0.0

    for i in tl.static_range(0, MAX_KV):
        # Validity conditions: i < candidate_max, i < num_kv_indices, kv_start + i < kv_end
        cond_i = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i

        # Load k_row and v_row for this idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Compute dot(q_sub, k_row) as scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * SM_SCALE

        # Update running max and sum for logsumexp
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(logits_scaled - m_new) * tl.where(cond_i, 1.0, 0.0)
        m = m_new

    # lse = log(s) + m (natural logsumexp)
    lse_val = tl.log(s) + m

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * NUM_QO_HEADS + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector for this query head
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < candidate_max) & (i < num_kv_indices) & (kv_start + i < kv_end)
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
    out_row_base = out_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM))
    out_vec_base = out_row_base + (h * HEAD_DIM)
    tl.store(out_vec_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.sm_scale = 1.0 / math.sqrt(128)
        self.max_kv = 256  # upper bound for loop; large enough for typical cases

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"

        total_q = q.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim = self.head_dim
        num_kv_heads = self.num_kv_heads

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # Squeeze the (1,) dimension: original k_cache has [num_pages, 1, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        # Allocate outputs
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)  # [total_q, 32, 128]
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)  # [total_q, 32]

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (qo_indptr.shape[0], total_q, num_qo_heads)
        _attention_one_triplet_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            out_f32, lse_f32,
            SM_SCALE=self.sm_scale if sm_scale is None else float(sm_scale),
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            MAX_KV=self.max_kv,
        )

        # Convert output to bfloat16 as in original
        output = out_f32.to(torch.bfloat16)
        return output, lse_f32


def run(*args):
    return ModelNew()(*args)
