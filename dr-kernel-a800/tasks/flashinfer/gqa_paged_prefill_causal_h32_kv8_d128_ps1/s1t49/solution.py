import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def attn_per_qh_kernel(
    q_ptr,            # *fp32, [total_q, num_qo_heads, head_dim]
    k_ptr,            # *fp32, [num_pages*num_kv_heads, head_dim]
    v_ptr,            # *fp32, [num_pages*num_kv_heads, head_dim]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *fp32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *fp32, [len_indptr-1, total_q, num_qo_heads]
    total_q: tl.constexpr,          # int32
    num_qo_heads: tl.constexpr,     # int32
    num_kv_heads: tl.constexpr,     # int32
    head_dim: tl.constexpr,         # int32
    GQA_RATIO: tl.constexpr,        # int32 = num_qo_heads // num_kv_heads
    SM_SCALE: tl.constexpr,         # fp32
    qo_indptr_ptr,    # *int32
    kv_indptr_ptr,    # *int32
    b: tl.constexpr,  # batch index 0..len_indptr-2
    q_idx: tl.constexpr,            # query token index within this batch
):
    # Load batch bounds and compute counts
    qo_start = tl.load(qo_indptr_ptr + b)      # int32
    qo_end = tl.load(qo_indptr_ptr + b + 1)    # int32
    kv_start = tl.load(kv_indptr_ptr + b)      # int32
    kv_end = tl.load(kv_indptr_ptr + b + 1)    # int32

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start

    # If no queries in this batch, skip
    if num_q_tokens == 0:
        return

    global_q_idx = q_idx + qo_start  # int32

    # Causal mask: max valid KV index
    delta = num_kv_tokens - num_q_tokens  # int32
    max_kv_idx = q_idx + 1 + delta
    if max_kv_idx > num_kv_tokens:
        max_kv_idx = num_kv_tokens

    # GQA mapping: each QO head maps to GQA_RATIO KV heads
    kv_head = h // GQA_RATIO  # int32

    # Load q vector for this head (fp32)
    q_row = q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    q_vec = tl.load(q_row + tl.arange(0, head_dim))  # [head_dim] fp32

    # Prepare logits buffer and output vector
    logits = tl.zeros([head_dim], dtype=tl.float32)
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    # Compute logits for j = 0..head_dim-1; mask j >= max_kv_idx with -large
    neg_large = -1e30  # fp32
    for j in range(0, head_dim):
        if j < max_kv_idx:
            k_id = tl.load(kv_indices_ptr + (kv_start + j))  # int32
            k_row = k_id * num_kv_heads + kv_head  # int32
            k_vec = tl.load(k_ptr + k_row * head_dim + tl.arange(0, head_dim))  # [head_dim] fp32
            dot = 0.0
            for d in range(0, head_dim):
                dot += q_vec[d] * k_vec[d]
            logits[j] = dot * SM_SCALE
        else:
            logits[j] = neg_large

    # Compute lse = logsumexp(logits) / ln(2)
    m = logits[0]
    for j in range(1, head_dim):
        if logits[j] > m:
            m = logits[j]
    sum_exp = 0.0
    for j in range(0, head_dim):
        sum_exp += tl.exp(logits[j] - m)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)

    # Compute attention and accumulate output
    for j in range(0, head_dim):
        if logits[j] > neg_large:  # valid kv index
            attn_j = tl.exp(logits[j] - m)  # fp32
            k_id = tl.load(kv_indices_ptr + (kv_start + j))  # int32
            k_row = k_id * num_kv_heads + kv_head  # int32
            v_vec_j = tl.load(v_ptr + k_row * head_dim + tl.arange(0, head_dim))  # [head_dim] fp32
            for d in range(0, head_dim):
                out_vec[d] += attn_j * v_vec_j[d]

    # Store results
    out_row = out_ptr + (b * num_q_tokens + q_idx) * (num_qo_heads * head_dim) + h * head_dim
    tl.store(out_row + tl.arange(0, head_dim), out_vec)

    lse_out = lse_ptr + (b * num_q_tokens + q_idx) * num_qo_heads + h
    tl.store(lse_out, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale=None):
        # Triton is required; ensure availability
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        device = q.device
        # Ensure contiguous and dtype for kernel
        q_fp32 = q.contiguous().to(torch.float32)
        # Flatten k_cache and v_cache along the 'pages' dimension: [num_pages, 1, num_kv_heads, head_dim] -> [num_pages*num_kv_heads, head_dim]
        k_fp32 = k_cache.contiguous().to(torch.float32).reshape(-1, self.num_kv_heads, self.head_dim)
        v_fp3


def run(*args):
    return ModelNew()(*args)
