import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,             # *float32, shape [total_q * num_qo_heads, head_dim]
    k_ptr,             # *float32, shape [num_pages * num_kv_heads, head_dim] (flattened)
    v_ptr,             # *float32, shape [num_pages * num_kv_heads, head_dim] (flattened)
    qo_indptr_ptr,     # *int32, shape [len_indptr]
    kv_indptr_ptr,     # *int32, shape [len_indptr]
    kv_indices_ptr,    # *int32, shape [num_kv_indices]
    output_ptr,        # *float32, shape [total_q * num_qo_heads, head_dim]
    output_lse_ptr,    # *float32, shape [total_q * num_qo_heads]
    sm_scale,          # float32
    q_start,           # int32
    q_end,             # int32
    kv_start,          # int32
    kv_end,            # int32
    head_dim: tl.constexpr,       # 128
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    gqa_ratio: tl.constexpr,      # 4 (num_qo_heads // num_kv_heads)
    MAX_Q_SEG: tl.constexpr,      # e.g., 128
    MAX_KV_SEG: tl.constexpr,     # e.g., 128
):
    b = tl.program_id(axis=0)  # segment index

    # Compute segment lengths (scalars, passed from host)
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Precompute 1/ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    ln2_inv = 1.0 / ln2

    # Iterate over query tokens in this segment with static loop and mask
    for q_i in range(0, MAX_Q_SEG):
        q_active = q_i < num_q_tokens_segment
        if not q_active:
            # no-op; static loop requires body, but mask prevents work
            pass
        global_q_idx = q_start + q_i

        # Base row offset in flattened q/output for this query
        row_offset = global_q_idx * num_qo_heads

        # Process each query head
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio  # GQA mapping

            # Compute logsumexp over KV tokens in this segment
            max_val = -float('inf')
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                kv_active = kk < num_kv_tokens
                if not kv_active:
                    # static loop; mask ensures no work
                    pass
                k_idx = kv_indices_ptr[kv_start + kk]

                # Load q vector for head h: q_ptr is [total_q * num_qo_heads, head_dim], row=row_offset
                q_off = row_offset * head_dim  # flatten to 1D: q_row offset
                q_vec = tl.load(q_ptr + q_off + tl.arange(0, head_dim), mask=kv_active, other=0.0)

                # Load k vector for this kv index and kv_head
                # k_ptr is flattened [num_pages * num_kv_heads, head_dim]
                k_off = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_off + tl.arange(0, head_dim), mask=kv_active, other=0.0)

                # Dot product
                # q_vec and k_vec are [head_dim]; compute scalar dot
                dot = tl.sum(q_vec * k_vec, axis=0)

                logit = dot * sm_scale

                # Update logsumexp stably
                new_max = tl.where(logit > max_val, logit, max_val)
                sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(logit - new_max)
                max_val = new_max

            # lse for this (global_q_idx, h)
            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            # Store lse at flattened index: row_offset * num_qo_heads + h
            tl.store(output_lse_ptr + row_offset * num_qo_heads + h, lse_val)

            # Compute causal attn window
            delta = num_kv_tokens - num_q_tokens_segment
            max_kv_idx = q_i + 1 + delta
            max_kv_idx = tl.minimum(max_kv_idx, num_kv_tokens)

            # Compute output vector for this head
            out_vec = tl.zeros([head_dim], dtype=tl.float32)

            for kk in range(0, MAX_KV_SEG):
                kv_active = kk < num_kv_tokens
                if not kv_active:
                    pass
                k_idx = kv_indices_ptr[kv_start + kk]

                q_off = row_offset * head_dim
                q_vec = tl.load(q_ptr + q_off + tl.arange(0, head_dim), mask=kv_active, other=0.0)

                k_off = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_off + tl.arange(0, head_dim), mask=kv_active, other=0.0)

                v_off = k_idx * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_off + tl.arange(0, head_dim), mask=kv_active, other=0.0)

                dot = tl.sum(q_vec * k_vec, axis=0)
                logit = dot * sm_scale

                # causal mask: set attn to 0 if kk >= max_kv_idx
                attn = tl.where(kv_active & (kk < max_kv_idx), tl.exp(logit), 0.0)

                out_vec += attn * v_vec

            # Store output vector: [row_offset, h, :]
            out_off = row_offset * num_qo_heads * head_dim + h * head_dim
            tl.store(output_ptr + out_off + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtype
        q = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]

        # Flatten k/v to [num_pages * num_kv_heads, head_dim]
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        k_flat = k_cache.view(num_pages * num_kv_heads, head_dim).contiguous()
        v_flat = v_cache.view(num_pages * num_kv_heads, head_dim).contiguous()

        total_q, num_qo_heads, _ = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"

        # Allocate outputs (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q, k_flat, v_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, output_lse,
            float(sm_scale),
            int(qo_indptr[0].item()), int(qo_indptr[-1].item()),
            int(kv_indptr[0].item()), int(kv_indptr[-1].item()),
            head_dim=128,
            num_qo_heads=32,
            num_kv_heads=8,
            gqa_ratio=4,
            MAX_Q_SEG=128,
            MAX_KV_SEG=128,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
