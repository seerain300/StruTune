import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_one_triplet_kernel(
    q_ptr,            # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,     # *float32, [num_pages, 8, 128]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, 32, 128]
    lse_ptr,          # *float32, [total_q, 32]
    SM_SCALE: tl.constexpr,    # scaling factor (float32)
    NUM_QO_HEADS: tl.constexpr,  # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    MAX_KV: tl.constexpr,         # e.g., 256
):
    # Program ids: each program handles one (b, q_idx, h)
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
    num_kv_indices = kv_end - kv_start

    # Global query index
    global_q_idx = qo_start + q_idx

    # GQA mapping: query head h uses KV head kv_head = h // 4
    kv_head = h // 4

    # Load q_sub[h] vector of length HEAD_DIM
    q_row_base = q_ptr + global_q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_vec = tl.load(q_row_base + tl.arange(0, HEAD_DIM))

    # Compute candidate_max and max_kv_idx based on causal-like masking logic
    candidate_max = q_idx + 1 + (num_kv_indices - num_q_tokens)
    max_kv_idx = tl.minimum(candidate_max, num_kv_indices)

    # First pass: compute logsumexp of scaled logits over valid kv positions
    m = -float("inf")
    s = 0.0
    for i in tl.static_range(0, MAX_KV):
        cond_i = (i < max_kv_idx) & (i < num_kv_indices) & (kv_start + i < kv_end)
        idx_i = kv_start + i
        # Compute pointer to k row for this idx_i and kv_head
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=cond_i, other=0.0)

        # Compute dot(q_sub[h], k_row) as scalar
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

    # Second pass: compute output vector for head h
    out_row_base = out_ptr + (global_q_idx * (NUM_QO_HEADS * HEAD_DIM))  # output index base
    out_vec_base = out_row_base + (h * HEAD_DIM)                       # head base
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
    tl.store(out_vec_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.sm_scale = 1.0 / math.sqrt(128.0)

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale=None):
        # Ensure tensors are on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # Squeeze the (1,) dimension to get [num_pages, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        total_q, num_qo_heads, _ = q_f32.shape
        num_pages, num_kv_heads, head_dim = k_cache_flat.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Allocate output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel
        grid = (len_indptr, total_q, num_qo_heads)
        _attention_one_triplet_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            SM_SCALE=self.sm_scale if sm_scale is None else float(sm_scale),
            NUM_QO_HEADS=self.num_qo_heads,
            NUM_KV_HEADS=self.num_kv_heads,
            HEAD_DIM=self.head_dim,
            MAX_KV=256,  # safe upper bound for kv_indices length
        )

        # Return output (float32) and lse (float32), as original code computes them
        # The original returns (output, lse), we keep that structure and dtypes.
        return output, lse

# Example helper functions to generate inputs (unchanged)
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
