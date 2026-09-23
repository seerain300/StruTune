import torch
import math

import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,                  # *f32, shape [total_q * num_qo_heads, head_dim]
    k_ptr, v_ptr,           # *f32, shape [num_pages * num_kv_heads, head_dim] (we pass flattened)
    qo_indptr_ptr,          # *i32, shape [len_indptr]
    kv_indptr_ptr,          # *i32, shape [len_indptr]
    kv_indices_ptr,         # *i32, shape [num_kv_indices]
    output_ptr,             # *f32, shape [total_q * num_qo_heads, head_dim]
    output_lse_ptr,         # *f32, shape [total_q * num_qo_heads]
    sm_scale,               # f32
    q_start, q_end,         # i32
    kv_start, kv_end,       # i32
    head_dim: tl.constexpr, # int
    num_qo_heads: tl.constexpr,  # 32
    num_kv_heads: tl.constexpr,  # 8
    gqa_ratio: tl.constexpr,     # 4
    MAX_Q_SEG: tl.constexpr,     # e.g., 128
    MAX_KV_SEG: tl.constexpr,    # e.g., 128
):
    # One program per segment
    b = tl.program_id(axis=0)

    # Segment lengths (scalars, passed from host)
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    ln2 = 0.6931471805599453  # math.log(2.0)
    ln2_inv = 1.0 / ln2

    # Iterate over query tokens in this segment with static loop and mask
    for q_i in range(0, MAX_Q_SEG):
        q_active = q_i < num_q_tokens_segment
        if not q_active:
            break  # static loop; mask ensures no work
        # Compute global query index
        global_q_idx = q_start + q_i
        row_offset = global_q_idx * num_qo_heads  # q_ptr is [total_q * num_qo_heads, head_dim]

        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio  # GQA mapping: 32 -> 8

            # Compute logsumexp over KV tokens in this segment using a static loop and mask
            max_val = -float("inf")
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                kv_active = kk < num_kv_tokens
                if not kv_active:
                    break  # static loop; mask ensures no work
                k_idx = kv_indices_ptr[kv_start + kk]  # i32

                # Load q vector for head h: q_ptr is [total_q * num_qo_heads, head_dim], row=row_offset
                q_row = tl.load(q_ptr + row_offset + tl.arange(0, head_dim))  # [head_dim]
                # Load k vector for this kv index and kv_head: k_ptr is [num_pages * num_kv_heads, head_dim]
                k_row = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))  # [head_dim]

                # Dot product (broadcast to 1x1 then reduce)
                dot = tl.sum(q_row * k_row, axis=0)  # scalar
                logit = dot * sm_scale

                # Update logsumexp
                new_max = tl.maximum(max_val, logit)
                sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(logit - new_max)
                max_val = new_max

            # lse = log(sum_exp) / ln(2)
            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            # Store lse for this (global_q_idx, h)
            tl.store(output_lse_ptr + row_offset + h, lse_val)

            # Compute causal attn window
            delta = num_kv_tokens - num_q_tokens_segment
            max_kv_idx = q_i + 1 + delta
            max_kv_idx = tl.minimum(max_kv_idx, num_kv_tokens)

            # Initialize output vector
            out_vec = tl.zeros([head_dim], dtype=tl.float32)

            # Compute output vector by applying softmax over valid kv indices
            for kk in range(0, MAX_KV_SEG):
                kv_active = kk < num_kv_tokens
                if not kv_active:
                    break  # static loop; mask ensures no work
                k_idx = kv_indices_ptr[kv_start + kk]
                q_row = tl.load(q_ptr + row_offset + tl.arange(0, head_dim))  # [head_dim]
                k_row = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))  # [head_dim]
                dot = tl.sum(q_row * k_row, axis=0)  # scalar
                logit = dot * sm_scale
                is_valid = kk < max_kv_idx
                attn = is_valid * tl.exp(logit - (max_val + tl.log(sum_exp)))  # softmax scaled by lse
                v_row = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + tl.arange(0, head_dim))  # [head_dim]
                out_vec += attn * v_row

            # Store output vector for this head
            tl.store(output_ptr + row_offset + h + tl.arange(0, head_dim), out_vec, mask=tl.full([1], True, tl.int1))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # q: [total_q, 32, 128], bfloat16; k_cache: [num_pages, 1, 8, 128], bfloat16; v_cache: [num_pages, 1, 8, 128], bfloat16
        # Convert to float32 for computation
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1, "k_cache/v_cache second dim must be 1"

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Ensure contiguity and float32
        q = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]

        # Flatten k_cache and v_cache to [num_pages, 8*128] (we use base pointer + kv_head * head_dim indexing directly)
        # Note: we will index by k_idx and kv_head, so we keep them as-is and compute addresses manually.

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)
        # Zero-initialize lse to match original behavior (lse initialized to -inf, then overwritten)
        output_lse.zero_()

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices,
            output, output_lse,
            sm_scale,
            # segment bounds as scalars
            qo_indptr[0].item(), qo_indptr[1].item(),
            kv_indptr[0].item(), kv_indptr[1].item(),
            head_dim,
            num_qo_heads, num_kv_heads, gqa_ratio,
            MAX_Q_SEG=128, MAX_KV_SEG=128,
            num_warps=4, num_stages=2,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
