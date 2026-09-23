import torch
import triton
import triton.language as tl


@triton.jit
def attn_single_qh_kernel(
    q_vec_ptr,      # *f32, [head_dim]
    k_rows_ptr,     # *f32, [num_k_rows, head_dim]
    v_rows_ptr,     # *f32, [num_k_rows, head_dim]
    out_row_ptr,    # *f32, [head_dim] - output for this (q_idx, h)
    lse_ptr,        # *f32, scalar lse for this (q_idx, h)
    head_dim: tl.constexpr,
    num_k_rows: tl.int32,
    sm_scale: tl.float32,
):
    # First pass: compute max(scaled) for numerical stability
    max_scaled = -float('inf')
    for j in range(0, num_k_rows):
        k_row = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        q_row = tl.load(q_vec_ptr + tl.arange(0, head_dim))
        dot = 0.0
        for d in range(0, head_dim):
            dot += q_row[d] * k_row[d]
        scaled = dot * sm_scale
        if scaled > max_scaled:
            max_scaled = scaled

    # Second pass: compute sumexp = sum(exp(scaled - max_scaled))
    sumexp = 0.0
    for j in range(0, num_k_rows):
        k_row = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        q_row = tl.load(q_vec_ptr + tl.arange(0, head_dim))
        dot = 0.0
        for d in range(0, head_dim):
            dot += q_row[d] * k_row[d]
        scaled = dot * sm_scale
        sumexp += tl.exp(scaled - max_scaled)

    # lse = logsumexp(scaled)/ln(2)
    lse_val = (max_scaled + tl.log(sumexp)) * 1.44269504  # 1/ln(2)

    # Store lse
    tl.store(lse_ptr, lse_val)

    # Third pass: compute attn and accumulate out_vec
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for j in range(0, num_k_rows):
        k_row = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        v_row = tl.load(v_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        q_row = tl.load(q_vec_ptr + tl.arange(0, head_dim))
        dot = 0.0
        for d in range(0, head_dim):
            dot += q_row[d] * k_row[d]
        scaled = dot * sm_scale
        attn_j = tl.exp(scaled - lse_val)  # softmax probability for this row
        for d in range(0, head_dim):
            out_vec[d] += attn_j * v_row[d]

    # Store out_vec
    for d in range(0, head_dim):
        tl.store(out_row_ptr + d, out_vec[d])


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
        k_cache_f32 = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v_cache_f32 = v_cache.squeeze(1).contiguous().to(torch.float32)

        total_q = q_f32.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128

        # Allocate outputs on CUDA
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device='cuda')
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device='cuda')

        # Number of batch intervals
        num_batches = qo_indptr.shape[0] - 1

        # Launch: one program per (b, q_idx, h). Host orchestrates loops.
        for b in range(num_batches):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start
            num_kv_tokens = kv_end - kv_start

            if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
                continue

            # Process each query token
            for q_idx in range(num_q_tokens):
                global_q_idx = qo_start + q_idx

                # Compute max_kv_idx considering causal mask
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_idx + 1 + delta
                if max_kv_idx > num_kv_tokens:
                    max_kv_idx = num_kv_tokens

                # If no valid KV for this query, skip
                if max_kv_idx <= 0:
                    continue

                # Process each query head
                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)  # 32 // 8 = 4
                    # Gather k_ids: [max_kv_idx]
                    k_ids = kv_indices[kv_start: kv_start + max_kv_idx].to(torch.int32).contiguous()

                    # Build k_rows and v_rows by indexing into k_cache_f32 and v_cache_f32
                    # Row index in flattened tensors: k_id * num_kv_heads + kv_head
                    k_rows = k_cache_f32[k_ids * num_kv_heads + kv_head]  # [max_kv_idx, 128]
                    v_rows = v_cache_f32[k_ids * num_kv_heads + kv_head]  # [max_kv_idx, 128]

                    # Load q vector for this head: q[global_q_idx, h, :]
                    q_vec = q_f32[global_q_idx, h, :].contiguous()

                    # Prepare output row pointer
                    out_row_ptr = output[global_q_idx, h, :].contiguous()
                    lse_ptr = lse[global_q_idx, h]

                    # Launch kernel for this (q_idx, h)
                    attn_single_qh_kernel[(1,)](
                        q_vec, k_rows, v_rows, out_row_ptr, lse_ptr,
                        head_dim=head_dim, num_k_rows=max_kv_idx, sm_scale=float(sm_scale),
                    )

        # Cast output to bfloat16 to match the original signature, keep lse in float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
