import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,              # *f32, [total_q * 32, 128]
    k_ptr,              # *f32, [num_pages, 8 * 128]
    v_ptr,              # *f32, [num_pages, 8 * 128]
    qo_indptr_ptr,      # *i32, [len_indptr]
    kv_indptr_ptr,      # *i32, [len_indptr]
    kv_indices_ptr,     # *i32, [num_kv_indices]
    output_ptr,         # *f32, [total_q * 32, 128]
    output_lse_ptr,     # *f32, [total_q * 32]
    sm_scale,           # f32 scalar
    q_start,            # i32 scalar
    q_end,              # i32 scalar
    kv_start,           # i32 scalar
    kv_end,             # i32 scalar
    total_q,            # i32
    num_qo_heads,       # i32, 32
    head_dim,           # i32, 128
    num_kv_heads,       # i32, 8
    gqa_ratio,          # i32, 4
    len_indptr,         # i32
    MAX_Q_SEG: tl.constexpr,   # e.g., 128
    MAX_KV_SEG: tl.constexpr,  # e.g., 128
):
    # One program per segment b
    b = tl.program_id(0)

    # Segment bounds
    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Constants
    ln2 = 0.6931471805599453  # math.log(2.0)
    ln2_inv = 1.0 / ln2

    # Iterate over query tokens in segment with static loop and mask
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            break
        global_q_idx = q_start + q_i

        # Iterate over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio  # GQA mapping

            # Compute numerically stable logsumexp over keys
            max_val = -float('inf')
            sum_exp = 0.0

            # Loop over kv tokens with mask
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + kv_start + kk).to(tl.int32)

                # Load q vector for this (global_q_idx, h): shape [head_dim]
                q_row_offset = global_q_idx * num_qo_heads + h
                q_vec = tl.load(q_ptr + q_row_offset * head_dim + tl.arange(0, head_dim)).to(tl.float32)

                # Load k vector for this kv index and kv head: shape [head_dim]
                k_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim)).to(tl.float32)

                # Dot product over head_dim
                prod = tl.sum(q_vec * k_vec, axis=0)

                scaled = prod * sm_scale
                # Update logsumexp in a numerically stable way
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            # Compute LSE = logsumexp(logits_scaled) / ln(2)
            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute attention softmax and output vector for this head
            # We need max_kv_idx for causal mask: min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            max_kv_idx = tl.minimum(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)

            # Accumulate output vector
            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + kv_start + kk).to(tl.int32)
                k_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                k_vec = tl.load(k_ptr + k_base + tl.arange(0, head_dim)).to(tl.float32)

                # Load corresponding v vector
                v_base = k_idx * num_kv_heads * head_dim + kv_head * head_dim
                v_vec = tl.load(v_ptr + v_base + tl.arange(0, head_dim)).to(tl.float32)

                prod = tl.sum((q_vec * k_vec), axis=0)  # scalar
                scaled = prod * sm_scale
                # Softmax over kk with causal mask
                active = kk < max_kv_idx
                # Compute softmax denominator
                exp_sum = 0.0
                for j in range(0, MAX_KV_SEG):
                    if j >= num_kv_tokens:
                        break
                    k_j = tl.load(kv_indices_ptr + kv_start + j).to(tl.int32)
                    k_base_j = k_j * num_kv_heads * head_dim + kv_head * head_dim
                    k_vec_j = tl.load(k_ptr + k_base_j + tl.arange(0, head_dim)).to(tl.float32)
                    v_base_j = k_j * num_kv_heads * head_dim + kv_head * head_dim
                    v_vec_j = tl.load(v_ptr + v_base_j + tl.arange(0, head_dim)).to(tl.float32)
                    prod_j = tl.sum(q_vec * k_vec_j, axis=0)
                    scaled_j = prod_j * sm_scale
                    exp_sum += tl.exp(scaled_j) * active

                attn_kk = tl.exp(scaled) / exp_sum
                out_vec += attn_kk * v_vec

            # Store output vector for this (global_q_idx, h)
            # Output is [total_q, 32, 128], we store flat: row = global_q_idx*32 + h, col = 0..127
            out_row_offset = global_q_idx * num_qo_heads + h
            tl.store(output_ptr + out_row_offset * head_dim + tl.arange(0, head_dim), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Ensure contiguity and float32
        q = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [num_pages, 1, 8, 128]

        # Flatten k_cache and v_cache to [num_pages, 8*128] for easy indexing
        k_flat = k_cache.view(num_pages, num_kv_heads * head_dim)  # [num_pages, 1024]
        v_flat = v_cache.view(num_pages, num_kv_heads * head_dim)  # [num_pages, 1024]

        # Output buffers (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        output_lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)

        attention_kernel[grid](
            q_ptr=q.view(total_q * num_qo_heads, head_dim),
            k_ptr=k_flat,              # [num_pages, 8*128]
            v_ptr=v_flat,              # [num_pages, 8*128]
            qo_indptr_ptr=qo_indptr,   # [len_indptr] int32
            kv_indptr_ptr=kv_indptr,   # [len_indptr] int32
            kv_indices_ptr=kv_indices, # [num_kv_indices] int32
            output_ptr=output.view(total_q * num_qo_heads, head_dim),
            output_lse_ptr=output_lse, # [total_q, 32] float32
            sm_scale=float(sm_scale),
            q_start=int(qo_indptr[0].item()),  # qo_indptr[0] = 0
            q_end=int(qo_indptr[1].item()),    # qo_indptr[1] gives first segment end
            kv_start=int(kv_indptr[0].item()),
            kv_end=int(kv_indptr[1].item()),
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            gqa_ratio=gqa_ratio,
            len_indptr=qo_indptr.shape[0],
            MAX_Q_SEG=128,
            MAX_KV_SEG=128,
        )

        return output, output_lse


def run(*args):
    return ModelNew()(*args)
