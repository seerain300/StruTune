import torch
import triton
import triton.language as tl


@triton.jit
def attention_qkv_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128] (host has flattened view; kernel uses k_id * num_kv_heads + kv_head)
    v_ptr,          # *f32, [num_pages, 8, 128]
    kv_indices_ptr, # *i32, [num_kv_indices]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    out_ptr,        # *f32, [total_q, 32, 128]
    lse_ptr,        # *f32, [total_q, 32]
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    sm_scale: tl.float32,
):
    b = tl.program_id(0)  # one program per batch element (interval)

    # Load qo_indptr[b], qo_indptr[b+1]
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    num_q_tokens = qo_end - qo_start

    # Load kv_indptr[b], kv_indptr[b+1]
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)
    num_kv_tokens = kv_end - kv_start

    # If intervals are invalid, skip
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    # Process each query token in this batch
    q_idx = 0
    while q_idx < num_q_tokens:
        global_q_idx = q_idx + qo_start

        # Compute max_kv_idx for this query (causal + delta)
        delta = num_kv_tokens - num_q_tokens
        max_kv_idx = q_idx + 1 + delta
        # If max_kv_idx exceeds num_kv_tokens, clamp to num_kv_tokens
        if max_kv_idx > num_kv_tokens:
            max_kv_idx = num_kv_tokens

        # If no valid KV entries for this query, skip (do nothing, then increment)
        if max_kv_idx <= 0:
            q_idx += 1
            continue  # 'continue' is allowed here; guarded

        # Process each query head
        h = 0
        while h < num_qo_heads:
            h_i = h
            kv_head = h_i // (num_qo_heads // num_kv_heads)  # 32 // 8 = 4

            # Load q vector for this head: q[global_q_idx, h, :]
            q_row = q_ptr + global_q_idx * (num_qo_heads * head_dim) + h_i * head_dim
            q_vec = tl.load(q_row + tl.arange(0, head_dim))  # [head_dim] f32

            # Build k_list_all and v_list_all: [max_kv_idx, head_dim]
            k_list_all = tl.zeros([max_kv_idx, head_dim], dtype=tl.float32)
            v_list_all = tl.zeros([max_kv_idx, head_dim], dtype=tl.float32)

            k_offset = kv_start
            j = 0
            while j < max_kv_idx:
                k_id = tl.load(kv_indices_ptr + k_offset + j)  # int32
                # row index in flattened k_ptr is k_id * num_kv_heads + kv_head
                row_k = k_id * num_kv_heads + kv_head
                k_vec = tl.load(k_ptr + row_k * head_dim + tl.arange(0, head_dim))
                k_list_all[j, :] = k_vec
                j += 1

            # Build v_list_all similarly
            j = 0
            while j < max_kv_idx:
                k_id = tl.load(kv_indices_ptr + k_offset + j)  # int32
                row_v = k_id * num_kv_heads + kv_head
                v_vec = tl.load(v_ptr + row_v * head_dim + tl.arange(0, head_dim))
                v_list_all[j, :] = v_vec
                j += 1

            # Compute logits = q_vec · k_list_all^T -> [max_kv_idx]
            logits = tl.zeros([max_kv_idx], dtype=tl.float32)
            for j in range(0, max_kv_idx):
                dot = 0.0
                for d in range(0, head_dim):
                    dot += q_vec[d] * k_list_all[j, d]
                logits[j] = dot

            # Scale logits
            scaled = logits * sm_scale

            # Compute lse = logsumexp(scaled) / ln(2)
            max_scaled = -float('inf')
            for j in range(0, max_kv_idx):
                if scaled[j] > max_scaled:
                    max_scaled = scaled[j]
            sumexp = 0.0
            for j in range(0, max_kv_idx):
                sumexp += tl.exp(scaled[j] - max_scaled)
            lse_val = (max_scaled + tl.log(sumexp)) * 1.44269504  # 1/ln(2)

            # Store lse to lse_ptr[b, h]
            lse_offset = b * num_qo_heads + h_i
            tl.store(lse_ptr + lse_offset, lse_val)

            # Compute attn = softmax(scaled)
            sumexp_shift = 0.0
            for j in range(0, max_kv_idx):
                sumexp_shift += tl.exp(scaled[j] - lse_val)
            attn = tl.zeros([max_kv_idx], dtype=tl.float32)
            for j in range(0, max_kv_idx):
                attn[j] = tl.exp(scaled[j] - lse_val) / sumexp_shift

            # Compute out_vec = attn · v_list_all -> [head_dim]
            out_row = out_ptr + global_q_idx * (num_qo_heads * head_dim) + h_i * head_dim
            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for d in range(0, head_dim):
                dot_v = 0.0
                for j in range(0, max_kv_idx):
                    dot_v += attn[j] * v_list_all[j, d]
                out_vec[d] = dot_v

            # Store out_vec
            for d in range(0, head_dim):
                tl.store(out_row + d, out_vec[d])

            h += 1

        q_idx += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        if q.device.type != 'cuda':
            q = q.to('cuda')
        if k_cache.device.type != 'cuda':
            k_cache = k_cache.to('cuda')
        if v_cache.device.type != 'cuda':
            v_cache = v_cache.to('cuda')
        if qo_indptr.device.type != 'cuda':
            qo_indptr = qo_indptr.to('cuda')
        if kv_indptr.device.type != 'cuda':
            kv_indptr = kv_indptr.to('cuda')
        if kv_indices.device.type != 'cuda':
            kv_indices = kv_indices.to('cuda')

        # Upcast to float32 for compute
        q_f32 = q.to(torch.float32)
        # Flatten k/v across the size-1 dimension
        k_cache_f32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
        v_cache_f32 = v_cache.squeeze(1).contiguous().to(torch.float32)

        total_q = q_f32.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        num_batches = qo_indptr.shape[0] - 1  # len_indptr - 1

        # Allocate outputs on CUDA
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device='cuda')
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device='cuda')

        # Launch Triton kernel: one program per batch element
        grid = (num_batches,)
        attention_qkv_kernel[grid](
            q_f32, k_cache_f32, v_cache_f32, kv_indices, qo_indptr, kv_indptr,
            output, lse,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            sm_scale=float(sm_scale),
        )

        # Cast output back to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
