import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_qkv_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128]
    v_ptr,          # *f32, [num_pages, 8, 128]
    kv_indices_ptr, # *i32, [num_kv_indices]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    out_ptr,        # *f32, [total_q, 32, 128]
    lse_ptr,        # *f32, [total_q, 32]
    total_q: tl.constexpr,           # total number of queries
    num_qo_heads: tl.constexpr,      # 32
    num_kv_heads: tl.constexpr,      # 8
    head_dim: tl.constexpr,          # 128
    gqa_ratio: tl.constexpr,         # 32 // 8 = 4
    sm_scale: tl.float32,
    qo_indptr_len: tl.constexpr,     # len_indptr
    kv_indptr_len: tl.constexpr,     # len_indptr
    num_q_tokens: tl.constexpr,      # number of queries in this batch element
    num_kv_tokens: tl.constexpr,     # number of KV tokens in this batch element
):
    # One program per batch element (b in [0..qo_indptr_len-2])
    b = tl.program_id(0)

    # Load qo_indptr[b], qo_indptr[b+1]
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)

    # Load kv_indptr[b], kv_indptr[b+1]
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Process each query index sequentially
    for q_idx in tl.static_range(0, num_q_tokens):
        global_q_idx = q_start + q_idx

        # Determine causal-bound max_kv_idx
        delta = num_kv_tokens - num_q_tokens
        if delta >= 0:
            max_kv_idx = q_idx + 1
        else:
            max_kv_idx = num_kv_tokens

        if max_kv_idx <= 0:
            continue

        # Process each query head
        for h in tl.static_range(0, num_qo_heads):
            # GQA mapping: 32 heads -> 8 kv heads, ratio 4
            kv_head = h // gqa_ratio

            # Load q vector for this head: q[global_q_idx, h, :]
            q_row_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            q_vec = tl.load(q_ptr + q_row_offset + tl.arange(0, head_dim))  # [head_dim] f32

            # Compute out_vec and accumulate lse
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            # We compute lse per element in the loop (single element at a time), since we process one q_idx
            # But we still need an accumulator for LSE across i
            lse_sum = tl.zeros((), dtype=tl.float32)
            exp_val = tl.zeros((), dtype=tl.float32)

            # Loop over i in [0..max_kv_idx-1]
            for i in tl.static_range(0, max_kv_idx):
                idx = kv_start + i
                if idx >= kv_end:
                    break
                k_page = tl.load(kv_indices_ptr + idx)  # flattened index into [num_pages]
                # k layout: [num_pages, num_kv_heads, head_dim]
                k_row_base = k_ptr + k_page * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec = tl.load(k_row_base + tl.arange(0, head_dim))  # [head_dim]
                # Logits: q_vec · k_vec
                logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
                scaled = logits * sm_scale
                # LSE accumulation: logsumexp over one element at a time
                # lse_sum += exp(scaled)
                lse_sum += tl.exp(scaled)
                # Softmax: softmax over a single element is just 1 since scaled is scalar, but we need attention weight
                # For softmax over a single element, attn = exp(scaled) / lse_sum
                attn = tl.exp(scaled) / lse_sum
                # Matvec: out += attn * v_vec
                v_row_base = v_ptr + k_page * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec = tl.load(v_row_base + tl.arange(0, head_dim))  # [head_dim]
                out_vec += attn * v_vec

            # Store output and LSE
            out_row_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            tl.store(out_ptr + out_row_offset + tl.arange(0, head_dim), out_vec)
            # LSE per head: lse_sum / ln(2)
            lse_row_offset = global_q_idx * num_qo_heads + h
            tl.store(lse_ptr + lse_row_offset, lse_sum * 1.4426950408889634)  # 1/ln(2)


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, _, num_kv_heads, _ = k_cache.shape
    device = q.device

    # Ensure CUDA tensors and float32 for computation
    q_f32 = q.to(torch.float32).contiguous()
    # Flatten num_pages dimension since original k_cache is [num_pages, 1, 8, 128]
    k_cache_f32 = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
    v_cache_f32 = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

    # Output tensors (float32, consistent with original behavior in run)
    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    len_indptr = qo_indptr.shape[0]
    grid = (len_indptr - 1,)

    # Launch one program per batch element. We compute num_q_tokens and num_kv_tokens on host and pass as constexpr.
    for b in range(len_indptr - 1):
        qo_start = int(qo_indptr[b].item())
        qo_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        num_q_tokens = max(0, qo_end - qo_start)
        num_kv_tokens = max(0, kv_end - kv_start)

        attention_qkv_kernel[grid](
            q_f32, k_cache_f32, v_cache_f32, kv_indices,
            qo_indptr, kv_indptr, output, lse,
            total_q=total_q,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            gqa_ratio=(num_qo_heads // num_kv_heads),  # 4
            sm_scale=sm_scale,
            qo_indptr_len=len_indptr,
            kv_indptr_len=len_indptr,
            num_q_tokens=num_q_tokens,
            num_kv_tokens=num_kv_tokens,
        )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if not q.is_cuda:
            q = q.cuda()
        if not k_cache.is_cuda:
            k_cache = k_cache.cuda()
        if not v_cache.is_cuda:
            v_cache = v_cache.cuda()
        if not qo_indptr.is_cuda:
            qo_indptr = qo_indptr.cuda()
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.cuda()
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.cuda()

        output, lse = run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
