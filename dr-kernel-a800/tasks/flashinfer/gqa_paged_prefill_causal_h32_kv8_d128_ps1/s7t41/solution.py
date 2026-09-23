import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,           # *f32, [total_q * 32, 128]
    k_ptr,           # *f32, [num_pages, 8*128]
    v_ptr,           # *f32, [num_pages, 8*128]
    qo_indptr_ptr,   # *i32, [len_indptr]
    kv_indptr_ptr,   # *i32, [len_indptr]
    kv_indices_ptr,  # *i32, [num_kv_indices]
    output_ptr,      # *f32, [total_q * 32, 128]
    output_lse_ptr,  # *f32, [total_q * 32]
    sm_scale,        # f32 scalar
    len_indptr,      # i32
    total_q,         # i32
    num_qo_heads,    # i32, 32
    head_dim,        # i32, 128
    num_kv_heads,    # i32, 8
    gqa_ratio,       # i32, 4
    MAX_Q_SEG: tl.constexpr,   # e.g., 128
    MAX_KV_SEG: tl.constexpr,  # e.g., 128
):
    # One program per segment
    b = tl.program_id(0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    ln2 = 0.6931471805599453  # math.log(2.0)
    ln2_inv = 1.0 / ln2

    # Iterate over query tokens in segment
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            break
        global_q_idx = q_start + q_i

        # Iterate over query heads (0..31)
        for h in range(0, 32):
            kv_head = h // gqa_ratio  # GQA mapping

            # Compute logsumexp over keys
            max_val = -float('inf')
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                # Load kv index
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)

                # Load k_vec for this kv entry and head: [head_dim]
                k_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim)).to(tl.float32)

                # Load q_vec for this (global_q_idx, h): [head_dim]
                q_row_offset = global_q_idx * num_qo_heads + h
                q_vec = tl.load(q_ptr + q_row_offset * head_dim + tl.arange(0, head_dim)).to(tl.float32)

                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale

                # Numerically stable update for logsumexp
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute softmax over keys and accumulate output
            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)

                k_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim)).to(tl.float32)

                q_row_offset = global_q_idx * num_qo_heads + h
                q_vec = tl.load(q_ptr + q_row_offset * head_dim + tl.arange(0, head_dim)).to(tl.float32)

                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale

                # Causal mask: zero out beyond max_kv_idx
                max_kv_idx = min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
                mask_active = kk < max_kv_idx
                scaled_masked = tl.where(mask_active, scaled, -float('inf'))

                # softmax = exp(scaled - max_val) / sum_exp
                softmax = tl.exp(scaled_masked - max_val) / sum_exp

                v_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_base + tl.arange(0, head_dim)).to(tl.float32)
                out_vec += softmax * v_vec

            # Store output for this (q_idx, h)
            q_row_offset = global_q_idx * num_qo_heads + h
            tl.store(output_ptr + q_row_offset * head_dim + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"

        # Ensure contiguity and float32
        q = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]

        # Flatten k_cache and v_cache to [num_pages, 8*128]
        num_pages = k_cache.shape[0]
        num_kv_heads = k_cache.shape[2]
        assert num_kv_heads == 8, "num_kv_heads must be 8"

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q_ptr=q.view(total_q * num_qo_heads, head_dim),
            k_ptr=k_cache.view(num_pages, num_kv_heads * head_dim),
            v_ptr=v_cache.view(num_pages, num_kv_heads * head_dim),
            qo_indptr_ptr=qo_indptr,
            kv_indptr_ptr=kv_indptr,
            kv_indices_ptr=kv_indices,
            output_ptr=output.view(total_q * num_qo_heads, head_dim),
            output_lse_ptr=output_lse,
            sm_scale=float(sm_scale),
            len_indptr=qo_indptr.shape[0],
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            gqa_ratio=(num_qo_heads // num_kv_heads),  # 4
            MAX_Q_SEG=128,
            MAX_KV_SEG=128,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
