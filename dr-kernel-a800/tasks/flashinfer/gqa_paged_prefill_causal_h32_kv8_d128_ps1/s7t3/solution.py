import math
import torch
import triton
import triton.language as tl


@triton.jit
def atencao_triton(
    q_ptr,           # *f32, [total_q, 32, 128]
    k_ptr,           # *f32, [num_pages, 8, 128]
    v_ptr,           # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,   # *i32, [len_indptr]
    kv_indptr_ptr,   # *i32, [len_indptr]
    kv_indices_ptr,  # *i32, [num_kv_indices]
    out_ptr,         # *f32, [total_q, 32, 128]
    lse_ptr,         # *f32, [total_q, 32]
    total_q,         # i32
    num_qo_heads,    # i32 (32)
    head_dim,        # i32 (128)
    num_pages,       # i32
    num_kv_heads,    # i32 (8)
    gqa_ratio,       # i32 (4)
    sm_scale,        # f32
    LN2,             # f32 = log(2)
    MAX_Q: tl.constexpr,             # upper bound on queries per segment
    MAX_KV_TOKENS: tl.constexpr,     # upper bound on kv tokens (>= num_pages)
):
    # One program processes one segment (b)
    b = tl.program_id(axis=0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b)         # i32
    q_end = tl.load(qo_indptr_ptr + b + 1)       # i32
    kv_start = tl.load(kv_indptr_ptr + b)        # i32
    kv_end = tl.load(kv_indptr_ptr + b + 1)      # i32

    # Number of valid keys for this segment
    num_kv_tokens = kv_end - kv_start            # i32

    # For each query index within this segment
    for q_idx in range(MAX_Q):
        if q_idx >= (q_end - q_start):
            break
        global_q_idx = q_start + q_idx           # i32

        # Loop over query heads
        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio             # 0..7

            # Compute logits vector for all keys in this segment
            logits = tl.zeros((MAX_KV_TOKENS,), dtype=tl.float32)
            for d in range(head_dim):
                # Load q component for this head
                q_off = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
                q_val = tl.load(q_ptr + q_off + d)  # scalar
                # Accumulate logits[kk] += q_val * k_ptr[kk, kv_head, d]
                for kk in range(MAX_KV_TOKENS):
                    valid_k = kk < num_kv_tokens
                    kv_idx = kv_start + kk
                    idx = tl.load(kv_indices_ptr + kv_idx)  # int32
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim + d
                    k_val = tl.load(k_ptr + k_off, mask=valid_k, other=0.0)  # scalar
                    logits[kk] += q_val * k_val

            # Scale logits by sm_scale
            logits_scaled = logits * sm_scale

            # Apply causal mask: max_kv_idx = min(q_idx + 1 + (num_kv_tokens - (q_end - q_start)), num_kv_tokens)
            num_q_tokens_segment = q_end - q_start
            delta = num_kv_tokens - num_q_tokens_segment
            delta = tl.maximum(delta, 0)
            max_kv_idx = tl.minimum(q_idx + 1 + delta, num_kv_tokens)
            # Mask logits: set to -inf beyond max_kv_idx
            for kk in range(MAX_KV_TOKENS):
                if kk >= max_kv_idx:
                    logits_scaled[kk] = -float("inf")

            # Compute LSE = logsumexp(logits_scaled) / LN2
            sum_exp = 0.0
            for kk in range(MAX_KV_TOKENS):
                sum_exp += tl.exp(logits_scaled[kk])
            lse = tl.log(sum_exp) / LN2

            # Compute attn = softmax(logits_scaled)
            attn = tl.zeros((MAX_KV_TOKENS,), dtype=tl.float32)
            for kk in range(MAX_KV_TOKENS):
                attn[kk] = tl.exp(logits_scaled[kk] - lse)
                if kk >= max_kv_idx:
                    attn[kk] = 0.0

            # Compute out_vec = attn @ v_ptr across keys for this head
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            for d in range(head_dim):
                # Sum over keys: attn[kk] * v_ptr[kk, kv_head, d]
                for kk in range(MAX_KV_TOKENS):
                    valid_k = kk < num_kv_tokens
                    kv_idx = kv_start + kk
                    idx = tl.load(kv_indices_ptr + kv_idx)
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim + d
                    v_val = tl.load(v_ptr + v_off, mask=valid_k, other=0.0)  # scalar
                    out_vec[d] += attn[kk] * v_val

            # Store output
            out_off = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            tl.store(out_ptr + out_off, out_vec)

            # Store LSE
            lse_off = global_q_idx * num_qo_heads + h
            tl.store(lse_ptr + lse_off, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity
        device = q.device
        assert device


def run(*args):
    return ModelNew()(*args)
