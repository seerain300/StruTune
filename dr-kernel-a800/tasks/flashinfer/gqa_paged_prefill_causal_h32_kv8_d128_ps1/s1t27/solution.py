import torch
import triton
import triton.language as tl


@triton.jit
def attention_fused_kernel(
    q_ptr,          # *f32, [total_q, 32, 128]
    k_ptr,          # *f32, [num_pages, 8, 128] (flattened view)
    v_ptr,          # *f32, [num_pages, 8, 128]
    kv_indices_ptr, # *i32, [num_kv_indices]
    qo_indptr_ptr,  # *i32, [len_indptr]
    kv_indptr_ptr,  # *i32, [len_indptr]
    out_ptr,        # *f32, [total_q, 32, 128]
    lse_ptr,        # *f32, [total_q, 32]
    total_q: tl.int32,
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    sm_scale: tl.float32,
):
    # One program per batch interval
    b = tl.program_id(0)

    # Load qo and kv interval bounds
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start

    # If intervals are invalid, skip
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    # Process each query token in this batch
    for q_idx in range(0, num_q_tokens):
        global_q_idx = q_idx + qo_start

        # Compute max_kv_idx for this query (delta + causal)
        delta = num_kv_tokens - num_q_tokens
        max_kv_idx = q_idx + 1 + delta
        if max_kv_idx > num_kv_tokens:
            max_kv_idx = num_kv_tokens

        # If no valid KV rows, skip (no inner loop work)
        if not (max_kv_idx > 0):
            continue  # Triton supports this 'continue'

        # For each query head h
        for h in range(0, num_qo_heads):
            # GQA mapping: each QO head uses a group of KV heads
            kv_head = h // (num_qo_heads // num_kv_heads)  # 32 // 8 = 4

            # Compute base address for q[global_q_idx, h, :]
            # q layout: [total_q, 32, 128] contiguous
            q_row_base = global_q_idx * (num_qo_heads * head_dim) + h * head_dim

            # Load q_row vector (head_dim elements)
            q_row_vec = tl.load(q_ptr + q_row_base + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)

            # Build k_all_rows [max_kv_idx, 128] and v_all_rows [max_kv_idx, 128]
            k_all_rows = tl.zeros([max_kv_idx, head_dim], dtype=tl.float32)
            v_all_rows = tl.zeros([max_kv_idx, head_dim], dtype=tl.float32)

            for j in range(0, max_kv_idx):
                kv_id = kv_start + j
                idx_k = tl.load(kv_indices_ptr + kv_id)  # i32
                k_addr = idx_k * (num_kv_heads * head_dim) + kv_head * head_dim
                k_vec_j = tl.load(k_ptr + k_addr + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)
                v_addr = idx_k * (num_kv_heads * head_dim) + kv_head * head_dim
                v_vec_j = tl.load(v_ptr + v_addr + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)

                k_all_rows[j, :] = k_vec_j
                v_all_rows[j, :] = v_vec_j

            # Compute logits = q_row_vec · k_all_rows^T -> [max_kv_idx]
            logits = tl.zeros([max_kv_idx], dtype=tl.float32)
            for j in range(0, max_kv_idx):
                dot = 0.0
                for d in range(0, head_dim):
                    dot += q_row_vec[d] * k_all_rows[j, d]
                logits[j] = dot

            # Scale and compute LSE in base-2
            scaled = logits * sm_scale
            sumexp = 0.0
            for j in range(0, max_kv_idx):
                sumexp += tl.exp(scaled[j])
            lse_val = tl.log(sumexp) / 0.6931471805599453  # 1 / ln(2)
            # Store LSE to lse[global_q_idx, h]
            tl.store(lse_ptr + global_q_idx * num_qo_heads + h, lse_val)

            # Compute attn = softmax(scaled)
            attn = tl.zeros([max_kv_idx], dtype=tl.float32)
            sumexp_shift = sumexp
            for j in range(0, max_kv_idx):
                attn[j] = tl.exp(scaled[j] - lse_val) / sumexp_shift

            # Compute out_vec = attn · v_all_rows -> [head_dim]
            out_row_base = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for d in range(0, head_dim):
                dot_v = 0.0
                for j in range(0, max_kv_idx):
                    dot_v += attn[j] * v_all_rows[j, d]
                out_vec[d] = dot_v

            # Store out_vec
            for d in range(0, head_dim):
                tl.store(out_ptr + out_row_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and dtype
        device = q.device
        if device.type != 'cuda':
            device = torch.device('cuda')
        q = q.to(device=device, dtype=torch.float32)
        k_cache = k_cache.to(device=device, dtype=torch.float32)
        v_cache = v_cache.to(device=device, dtype=torch.float32)
        qo_indptr = qo_indptr.to(device=device, dtype=torch.int32)
        kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.to(device=device, dtype=torch.int32)

        # Flatten k/v across the size-1 dimension
        k_cache_flat = k_cache.squeeze(1).contiguous()
        v_cache_flat = v_cache.squeeze(1).contiguous()

        total_q = q.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch interval
        len_indptr = qo_indptr.shape[0]
        grid = (len_indptr - 1,)
        attention_fused_kernel[grid](
            q, k_cache_flat, v_cache_flat, kv_indices, qo_indptr, kv_indptr, output, lse,
            total_q, num_qo_heads, num_kv_heads, head_dim, sm_scale,
            num_warps=1, num_stages=1
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
