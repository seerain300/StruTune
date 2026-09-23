import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_segment_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128]
    v_ptr,          # *f32, [num_pages, 8, 128]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    kv_indices_ptr, # *i32, [num_kv_indices]
    output_ptr,     # *f32, [total_q, 32, 128]
    output_lse_ptr, # *f32, [total_q, 32]
    sm_scale,       # f32 scalar
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
    num_qo_heads: tl.constexpr,  # 32
    num_kv_heads: tl.constexpr,  # 8
    head_dim: tl.constexpr,      # 128
    gqa_ratio: tl.constexpr,     # 4
):
    b = tl.program_id(0)

    # Load segment bounds
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # Vectorize over query index up to MAX_Q_SEG, mask q_i >= num_q_tokens_segment
    for q_i in range(0, MAX_Q_SEG):
        if q_i >= num_q_tokens_segment:
            continue
        global_q_idx = q_start + q_i

        # Process each query head h
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Compute logsumexp over keys for this (q_i, h)
            max_val = -float('inf')
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                # Load q[h] vector
                q_vec = tl.load(q_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                # Load k_vec for this key index
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)
                k_vec = tl.load(k_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                prod = tl.sum(q_vec * k_vec, axis=0)  # scalar
                scaled = prod * sm_scale
                # Numerically stable update
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            # lse = logsumexp(scaled) / ln(2)
            ln2 = 0.6931471805599453  # math.log(2.0)
            lse_val = (max_val + tl.log(sum_exp)) / ln2
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute output vector: attn * v
            # Recompute max_val/sum_exp for softmax (could reuse, but compute again to keep code simple)
            max_val = -float('inf')
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                q_vec = tl.load(q_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)
                k_vec = tl.load(k_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                prod = tl.sum(q_vec * k_vec, axis=0)  # scalar
                scaled = prod * sm_scale
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            # Compute attn under causal mask
            # max_kv_idx = min(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
            max_kv_idx = tl.minimum(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)

            # Now write out = attn * v for each kk up to MAX_KV_SEG, masking kk < max_kv_idx
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                q_vec = tl.load(q_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk)).to(tl.int32)
                k_vec = tl.load(k_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                prod = tl.sum(q_vec * k_vec, axis=0)  # scalar
                scaled = prod * sm_scale
                # Softmax contribution: exp(scaled - max_val) / sum_exp
                if kk < max_kv_idx:
                    attn_val = tl.exp(scaled - max_val) / sum_exp
                else:
                    attn_val = 0.0

                v_vec = tl.load(v_ptr + k_idx * num_kv_heads * head_dim + kv_head * head_dim + tl.arange(0, head_dim)).to(tl.float32)
                out_vec += attn_val * v_vec

            # Store output for this (global_q_idx, h)
            tl.store(output_ptr + global_q_idx * num_qo_heads * head_dim + h * head_dim + tl.arange(0, head_dim), out_vec, mask=True)

# Host forward: Triton-only computation, no torch math
class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        device = q.device

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]
        # Constants (as in original)
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert len_indptr >= 2

        # Convert to float32 for kernel computation
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.to(torch.float32)
        v_cache_f32 = v_cache.to(torch.float32)
        qo_indptr_i32 = qo_indptr.to(torch.int32)
        kv_indptr_i32 = kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices.to(torch.int32)

        # Allocate outputs (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per segment
        grid = (len_indptr - 1,)
        attn_segment_kernel[grid](
            q_f32, k_cache_f32, v_cache_f32,
            qo_indptr_i32, kv_indptr_i32, kv_indices_i32,
            output, lse,
            float(sm_scale),
            MAX_Q_SEG=4096,    # upper bound for q tokens per segment
            MAX_KV_SEG=4096,   # upper bound for kv tokens per segment
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            gqa_ratio=4,
            num_warps=4,
            num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
