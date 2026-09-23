import torch
import triton
import triton.language as tl

# Triton kernel: one program per (b, q_idx, h)
@triton.jit
def attn_per_qh_kernel(
    q_ptr,               # *f32, shape [total_q * num_qo_heads, head_dim]
    k_ptr,               # *f32, shape [num_pages * num_kv_heads, head_dim]
    v_ptr,               # *f32, shape [num_pages * num_kv_heads, head_dim]
    out_ptr,             # *f32, shape [total_q * num_qo_heads, head_dim]
    lse_ptr,             # *f32, shape [total_q * num_qo_heads]
    kv_indices_ptr,      # *i32, shape [num_kv_tokens]
    # scalar params
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    sm_scale: tl.constexpr,       # 1/sqrt(128)
    b: tl.int32,                  # batch index
    qo_start: tl.int32,           # qo_indptr[b]
    num_q_tokens: tl.int32,       # qo_indptr[b+1] - qo_indptr[b]
    kv_start: tl.int32,           # kv_indptr[b]
    num_kv_tokens: tl.int32,      # kv_indptr[b+1] - kv_indptr[b]
    q_idx: tl.int32,              # query token index within this batch
    h: tl.int32,                  # query head index
):
    # Compute max_kv_idx following original code:
    # delta = num_kv_tokens - num_q_tokens
    # max_kv_idx = q_idx + 1 + delta, clamped to num_kv_tokens
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx = q_idx + 1 + delta
    if max_kv_idx > num_kv_tokens:
        max_kv_idx = num_kv_tokens

    # If max_kv_idx <= 0, host will not launch this program. We still guard with masks.

    # Global q index
    global_q_idx = q_idx + qo_start

    # GQA mapping: kv_head = h // (num_qo_heads // num_kv_heads) = h // 4
    gqa_ratio = num_qo_heads // num_kv_heads  # 4
    kv_head = h // gqa_ratio

    # Load q vector for this head: q[global_q_idx, h, :]
    q_row_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    q_vec = tl.load(q_ptr + q_row_offset + tl.arange(0, head_dim))

    # Output vector and lse scalar for this (q_idx, h)
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    lse_val = tl.zeros((), dtype=tl.float32)

    # Prepare buffer for scaled_logits[j] over j in [0, max_kv_idx); pad with -inf for j >= max_kv_idx
    scaled_logits = tl.zeros([head_dim], dtype=tl.float32)
    neg_large = -1e30
    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)
        k_row = k_id * num_kv_heads + kv_head
        k_vec = tl.load(k_ptr + k_row * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
        v_vec = tl.load(v_ptr + k_row * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)

        dot = 0.0
        d = 0
        while d < head_dim:
            dot += q_vec[d] * k_vec[d]
            d += 1

        scaled_logits[j] = dot * sm_scale
        j += 1

    # Set remaining entries to -inf for logsumexp masking
    for j in range(max_kv_idx, head_dim):
        scaled_logits[j] = neg_large

    # Compute lse = logsumexp(scaled_logits) / ln(2)
    m = scaled_logits[0]
    j = 1
    while j < head_dim:
        if scaled_logits[j] > m:
            m = scaled_logits[j]
        j += 1

    sum_exp = 0.0
    j = 0
    while j < head_dim:
        sum_exp += tl.exp(scaled_logits[j] - m)
        j += 1

    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)

    # Compute attn and accumulate output
    j = 0
    while j < max_kv_idx:
        scaled_j = scaled_logits[j]
        attn_j = tl.exp(scaled_j - m)
        # Load V for this j
        k_id_j = tl.load(kv_indices_ptr + kv_start + j)
        k_row_j = k_id_j * num_kv_heads + kv_head
        v_vec_j = tl.load(v_ptr + k_row_j * head_dim + tl.arange(0, head_dim), mask=True, other=0.0)
        # Accumulate out_vec += attn_j * v_vec_j
        d = 0
        while d < head_dim:
            out_vec[d] += attn_j * v_vec_j[d]
            d += 1
        j += 1

    # Store results
    out_row_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    tl.store(out_ptr + out_row_offset + tl.arange(0, head_dim), out_vec)
    tl.store(lse_ptr + (b * num_q_tokens + q_idx) * num_qo_heads + h, lse_val)

# ModelNew: uses Triton kernels, no torch ops in forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure shapes and dtypes
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        assert qo_indptr


def run(*args):
    return ModelNew()(*args)
