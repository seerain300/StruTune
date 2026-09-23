import math
import torch
import triton
import triton.language as tl

# Triton kernel: compute attention for a single (b, q_idx, h) triple.
# It iterates j over the valid kv indices, loads k_row and v_row, computes scaled dot, accumulates lse,
# then recomputes attn and accumulates output vector. No "continue"; we use if and while with bounds.
@triton.jit
def attn_single_kernel(
    q_vec_ptr,                        # *float32, points to a vector of length head_dim
    k_cache_flat_ptr, v_cache_flat_ptr,  # *float32, each has shape [num_pages * num_kv_heads, head_dim]
    kv_indices_ptr,                  # *int32, length num_kv_indices
    out_ptr,                         # *float32, output[b, h, :] vector, length head_dim
    lse_ptr,                         # *float32, scalar lse for (b, h)
    num_kv_tokens: tl.int32,
    kv_start: tl.int32,
    total_q: tl.int32,               # not used directly, but passed for signature symmetry
    num_q_tokens: tl.int32,          # not used directly
    delta: tl.int32,                 # num_kv_tokens - num_q_tokens
    max_kv_idx: tl.int32,            # min(q_idx + 1 + delta, num_kv_tokens)
    head_dim: tl.constexpr,          # e.g., 128
    num_kv_heads: tl.int32,          # e.g., 8
    gqa_ratio: tl.int32,             # e.g., 4
    sm_scale: tl.float32,
):
    # Running max and sum for LSE
    m = -float('inf')
    sumexp = 0.0

    # First pass: compute LSE
    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)  # int32
        kv_head = h // gqa_ratio  # GQA mapping: each qo head uses 8*4 = 32 kv heads -> head // 4
        # Row id in flattened cache
        row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
        # Load k_row and q_vec (both length head_dim)
        k_row = tl.load(k_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                 # [head_dim]
        # Dot product
        dot = tl.sum(q_vec * k_row)
        scaled = dot * sm_scale
        m_new = tl.maximum(m, scaled)
        # Rescale sumexp
        sumexp = sumexp * tl.exp(m - m_new) + tl.exp(scaled - m_new)
        m = m_new
        j += 1

    # LSE per original: logsumexp(scaled) / ln(2)
    lse_val = (m + tl.log(sumexp)) / math.log(2.0)

    # Second pass: compute output vector
    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)
        kv_head = h // gqa_ratio
        row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
        k_row = tl.load(k_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                 # [head_dim]
        dot = tl.sum(q_vec * k_row)
        scaled = dot * sm_scale
        attn = tl.exp(scaled - lse_val)

        v_row = tl.load(v_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim]
        out_vec = attn * v_row

        # Store out_vec into out_ptr (vectorized store). Triton supports vectorized store to a pointer.
        offsets = tl.arange(0, head_dim)
        tl.store(out_ptr + offsets, out_vec, mask=offsets < head_dim)

        j += 1

    # Store lse scalar
    tl.store(lse_ptr, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and float32 compute
        device = q.device
        if device.type != 'cuda':
            q = q.to('cuda')
            k_cache = k_cache.to('cuda')
            v_cache = v_cache.to('cuda')
            qo_indptr = qo_indptr.to('cuda')
            kv_indptr = kv_indptr.to('cuda')
            kv_indices = kv_indices.to('cuda')

        q_f32 = q.to(torch.float32)
        # Flatten k_cache and v_cache since page_size == 1
        k_cache_flat = k_cache.squeeze(1).contiguous().to(torch.float32)
        v_cache_flat = v_cache.squeeze(1).contiguous().to(torch.float32)

        total_q, num_qo_heads, head_dim = q_f32.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Assertions
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert k_cache.shape[1] == 1 and v_cache.shape[1] == 1
        assert total_q == int(qo_indptr[-1].item())

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

        # Host loops over segments b, queries q_idx, and heads h
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            if num_q_tokens <= 0:
                continue

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                if max_kv_idx == 0:
                    continue

                for h in range(num_qo_heads):
                    gqa_ratio = num_qo_heads // num_kv_heads  # 4
                    kv_head = h // gqa_ratio

                    # Prepare q_vec pointer: q_f32[global_q_idx, h, :]
                    q_vec = q_f32[global_q_idx, h]  # [head_dim]
                    q_vec_ptr = q_vec  # Triton expects a pointer; torch.Tensor works as pointer in Triton

                    # Allocate output vector
                    out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)

                    # Launch Triton kernel for this (b, q_idx, h)
                    grid = (1,)
                    attn_single_kernel[grid](
                        q_vec_ptr,
                        k_cache_flat, v_cache_flat,
                        kv_indices,
                        out_vec,
                        lse[b, h],
                        num_kv_tokens,
                        kv_start,
                        total_q,
                        num_q_tokens,
                        delta,
                        max_kv_idx,
                        head_dim,
                        num_kv_heads,
                        gqa_ratio,
                        sm_scale,
                    )

                    # Store result into output
                    output[global_q_idx, h] = out_vec

        # Return output in bfloat16 (as in original), and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
