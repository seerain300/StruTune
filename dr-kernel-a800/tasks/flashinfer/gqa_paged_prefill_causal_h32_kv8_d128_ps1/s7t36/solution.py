import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,                 # *f32, [total_q, 32, 128]
    k_ptr,                 # *f32, [num_pages, 8, 128], flattened to [num_pages*8, 128]
    v_ptr,                 # *f32, [num_pages, 8, 128], flattened to [num_pages*8, 128]
    output_ptr,            # *f32, [total_q, 32, 128]
    output_lse_ptr,        # *f32, [total_q, 32]
    qo_indptr_b,           # i32, qo_indptr[b]
    qo_indptr_b1,          # i32, qo_indptr[b+1]
    kv_indptr_b,           # i32, kv_indptr[b]
    kv_indptr_b1,          # i32, kv_indptr[b+1]
    head_dim,              # i32, 128
    num_qo_heads,          # i32, 32
    num_kv_heads,          # i32, 8
    gqa_ratio,             # i32, 4
    sm_scale,              # f32
    ln2_inv,               # f32
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    # One program per segment
    b = tl.program_id(0)

    num_q_tokens_segment = qo_indptr_b1 - qo_indptr_b
    num_kv_tokens = kv_indptr_b1 - kv_indptr_b

    dim = tl.arange(0, head_dim)

    # Iterate over query tokens in this segment
    for q_i in range(0, MAX_Q_SEG):
        q_valid = q_i < num_q_tokens_segment
        global_q_idx = qo_indptr_b + q_i
        if not q_valid:
            continue

        # Iterate over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Load q[h] vector
            q_vec = tl.load(q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim)

            # Compute logsumexp over keys (numerically stable)
            max_val = -float("inf")
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indptr_b + kk)  # int32
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute softmax-weighted output (causal mask)
            delta = num_kv_tokens - num_q_tokens_segment
            candidate = q_i + 1 + delta
            max_kv_idx = candidate if candidate > 0 else 0
            max_kv_idx = max_kv_idx if max_kv_idx <= num_kv_tokens else num_kv_tokens

            total_prob = 0.0
            out_vec = tl.zeros([head_dim], dtype=tl.float32)

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                if kk >= max_kv_idx:
                    continue
                k_idx = tl.load(kv_indptr_b + kk)  # int32
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                attn = tl.exp(scaled - lse_val)
                total_prob += attn

                v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                out_vec += attn * v_vec

            # Normalize by total_prob (softmax sum)
            out_vec = out_vec / total_prob
            tl.store(output_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are CUDA and float32
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA tensors"
        device = q.device

        # Cast inputs to float32 for stable math in Triton
        q_f32 = q.contiguous().to(torch.float32)  # [total_q, 32, 128]
        # Flatten k/v to [num_pages * 8, 128] contiguous
        k_f32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128] -> [num_pages*8, 128]
        v_f32 = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128] -> [num_pages*8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        # num_kv_heads is implied by GQA (8), gqa_ratio = 4 (32/8)
        gqa_ratio = 4

        # Output tensors (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        ln2_inv = 1.0 / math.log(2.0)

        # Launch Triton kernel: one program per segment
        grid = (qo_indptr.shape[0] - 1,)
        # Choose block sizes (compile-time constexpr). Masks handle actual sizes.
        MAX_Q_SEG = 256   # upper bound for queries per segment
        MAX_KV_SEG = 512  # upper bound for kv tokens

        attention_kernel[grid](
            q_f32, k_f32, v_f32, output, lse,
            qo_indptr[0].item(), qo_indptr[1].item(),
            kv_indptr[0].item(), kv_indptr[1].item(),
            head_dim, num_qo_heads, 8, gqa_ratio, sm_scale, ln2_inv,
            MAX_Q_SEG=MAX_Q_SEG, MAX_KV_SEG=MAX_KV_SEG,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
