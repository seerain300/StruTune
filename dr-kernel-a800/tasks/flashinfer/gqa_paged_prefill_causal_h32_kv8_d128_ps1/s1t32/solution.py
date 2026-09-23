import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_qh_kernel(
    q_vec_ptr,                 # *f32, [head_dim]
    k_rows_ptr,                # *f32, [num_k_rows, head_dim] contiguous
    v_rows_ptr,                # *f32, [num_k_rows, head_dim] contiguous
    out_row_ptr,               # *f32, [head_dim]
    lse_ptr,                   # *f32, scalar
    head_dim: tl.constexpr,    # e.g., 128
    num_k_rows: tl.int32,      # number of KV rows to consider
    sm_scale: tl.float32,      # scaling factor
):
    # Running max and sumexp for stable logsumexp
    m = -float('inf')
    sumexp = 0.0
    m_old = -float('inf')

    # First pass: compute logsumexp over all j in [0, num_k_rows)
    j = 0
    while j < num_k_rows:
        k_row = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        v_row = tl.load(v_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                  # [head_dim]

        dot = 0.0
        for d in range(0, head_dim):
            dot += q_vec[d] * k_row[d]

        scaled = dot * sm_scale
        new_m = tl.maximum(m, scaled)
        # rescale sumexp to new_m
        sumexp = sumexp * tl.exp(m_old - new_m) + tl.exp(scaled - new_m)
        m_old = new_m
        m = new_m

        j += 1

    # lse = (m + log(sumexp)) / ln(2)
    ln2_inv = 1.44269504  # 1 / ln(2)
    lse_val = (m + tl.log(sumexp)) * ln2_inv
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output vector by accumulating attn * v_row for each j
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    j = 0
    while j < num_k_rows:
        k_row = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        v_row = tl.load(v_rows_ptr + j * head_dim + tl.arange(0, head_dim))  # [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                  # [head_dim]

        dot = 0.0
        for d in range(0, head_dim):
            dot += q_vec[d] * k_row[d]

        scaled = dot * sm_scale
        attn = tl.exp(scaled - lse_val)
        # Accumulate out_vec += attn * v_row
        for d in range(0, head_dim):
            out_vec[d] += attn * v_row[d]

        j += 1

    # Store the output row
    for d in range(0, head_dim):
        tl.store(out_row_ptr + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA and cast to float32 for compute
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

        q = q.to(torch.float32)
        # Flatten k_cache and v_cache along num_pages * num_kv_heads
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        # k_cache_flat: [num_pages*num_kv_heads, head_dim]
        k_cache_flat = k_cache.reshape(num_pages * num_kv_heads, head_dim).to(torch.float32)
        v_cache_flat = v_cache.reshape(num_pages * num_kv_heads, head_dim).to(torch.float32)

        total_q, num_qo_heads, _ = q.shape
        assert num_qo_heads == 32
        assert head_dim == 128
        assert num_kv_heads == 8

        device = q.device
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            if num_q_tokens <= 0:
                continue

            num_kv_tokens = kv_end - kv_start
            if num_kv_tokens <= 0:
                continue

            delta = num_kv_tokens - num_q_tokens

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                # max_kv_idx follows the original code's logic
                max_kv_idx = max(0, min(q_idx + 1 + delta, num_kv_tokens))
                if max_kv_idx == 0:
                    continue

                # Build k_rows_flat and v_rows_flat for this batch element and q_idx
                k_rows_flat = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=device)
                v_rows_flat = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=device)

                # For each j in [0, max_kv_idx), k_id = kv_indices[kv_start + j]
                for j in range(max_kv_idx):
                    k_id = int(kv_indices[kv_start + j].item())
                    kv_head = (q_idx % num_qo_heads) // gqa_ratio  # head index doesn't matter here; GQA uses h
                    # In original, kv_head = h // 4; but we already know b. For k_rows, kv_head depends on h.
                    # We need kv_head per head. We can compute kv_head from h when launching kernel.
                    # So we prepare k_rows_flat/v_rows_flat per head inside kernel by indexing using h.
                    # For now, compute for each head separately.

                # Instead of building per-h subsets here, we will construct them inside each head loop below.
                # To avoid nested host loops, we handle per head by launching the kernel per (b, q_idx, h).

        # Launch per (b, q_idx, h):
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            if num_q_tokens <= 0:
                continue

            num_kv_tokens = kv_end - kv_start
            if num_kv_tokens <= 0:
                continue

            delta = num_kv_tokens - num_q_tokens

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                max_kv_idx = max(0, min(q_idx + 1 + delta, num_kv_tokens))
                if max_kv_idx == 0:
                    continue

                # For each head h
                for h in range(num_qo_heads):
                    # Compute kv_head for GQA
                    kv_head = h // gqa_ratio

                    # Build k_rows_flat and v_rows_flat for this (b, q_idx, h)
                    k_rows_flat = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=device)
                    v_rows_flat = torch.empty((max_kv_idx, head_dim), dtype=torch.float32, device=device)

                    for j in range(max_kv_idx):
                        k_id = int(kv_indices[kv_start + j].item())
                        # Flat index in k_cache_flat: row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
                        row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
                        k_rows_flat[j, :] = k_cache_flat[row_id: row_id + head_dim]
                        v_rows_flat[j, :] = v_cache_flat[row_id: row_id + head_dim]

                    # q vector for this head
                    q_row = q[global_q_idx, h, :].contiguous().to(torch.float32)  # [head_dim]
                    out_row = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    lse_scalar = torch.empty((), dtype=torch.float32, device=device)

                    # Launch kernel
                    attention_single_qh_kernel[(1,)](
                        q_row, k_rows_flat, v_rows_flat, out_row, lse_scalar,
                        head_dim=128, num_k_rows=max_kv_idx, sm_scale=sm_scale
                    )

                    # Store results
                    output[global_q_idx, h, :] = out_row
                    lse[global_q_idx, h] = lse_scalar[0]

        # Return bfloat16 output and float32 lse
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
