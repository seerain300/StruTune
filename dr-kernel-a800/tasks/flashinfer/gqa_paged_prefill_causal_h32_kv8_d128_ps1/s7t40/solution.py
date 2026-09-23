import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_lse_kernel(
    q_ptr,            # *f32, [T, 128], T = total_q * 32
    k_ptr,            # *f32, [num_pages, 8, 128] flattened to [num_pages, 8*128]
    kv_indices_ptr,   # *i32, [num_kv_indices]
    qo_indptr_ptr,    # *i32, [len_indptr]
    kv_indptr_ptr,    # *i32, [len_indptr]
    output_lse_ptr,   # *f32, [total_q, 32]
    sm_scale,         # f32 scalar
    head_dim: tl.constexpr,      # 128
    num_qo_heads: tl.constexpr,  # 32
    num_kv_heads: tl.constexpr,  # 8
    gqa_ratio: tl.constexpr,     # 4
    MAX_Q_SEG: tl.constexpr,     # upper bound for query tokens per segment
    MAX_KV_SEG: tl.constexpr,    # upper bound for kv tokens per segment
):
    # One program per segment
    b = tl.program_id(0)

    # Load segment bounds (scalars)
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    # Number of tokens in this segment
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Constants
    ln2 = 0.6931471805599453  # natural log of 2
    ln2_inv = 1.0 / ln2

    # Iterate over queries in this segment
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            break
        global_q_idx = q_start + q_i

        # Iterate over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Compute logsumexp over key list (scaled) with causal mask
            max_val = -float("inf")
            sum_exp = 0.0

            # Iterate keys up to MAX_KV_SEG with mask
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                # Load index into kv_indices
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)

                # Load k_vec for this kv page and head: [128]
                # k_ptr is [num_pages, 8, 128] flattened to [num_pages, 8*128]
                k_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim)).to(tl.float32)

                # Load q_vec for this (global_q_idx, h): [128]
                # q_ptr is [T, 128], T = total_q * 32
                q_row_offset = global_q_idx * num_qo_heads + h
                q_vec = tl.load(q_ptr + q_row_offset * head_dim + tl.arange(0, head_dim)).to(tl.float32)

                # Dot product across head_dim
                prod = tl.sum(q_vec * k_vec, axis=0)

                scaled = prod * sm_scale

                # Update logsumexp in a numerically stable way
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            # Compute final lse / ln(2)
            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are contiguous and float32
        total_q, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)


def run(*args):
    return ModelNew()(*args)
