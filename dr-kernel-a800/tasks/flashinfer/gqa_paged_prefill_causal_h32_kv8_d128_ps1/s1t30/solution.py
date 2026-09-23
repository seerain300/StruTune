import torch
import triton
import triton.language as tl


@triton.jit
def attention_bh_kernel(
    q_vec_ptr,       # *f32, [head_dim] pointer to q[global_q_idx, h, :]
    k_rows_ptr,      # *f32, [num_k_rows, head_dim]
    v_rows_ptr,      # *f32, [num_k_rows, head_dim]
    num_k_rows: tl.int32,
    head_dim: tl.constexpr,
    sm_scale: tl.float32,
    out_row_ptr,     # *f32, [head_dim] pointer to output[global_q_idx, h, :]
    lse_ptr,         # *f32, pointer to lse[global_q_idx, h]
):
    # Load q vector
    q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))  # [head_dim] f32

    # Build k_list_all and v_list_all: [num_k_rows, head_dim]
    k_list_all = tl.zeros([num_k_rows, head_dim], dtype=tl.float32)
    v_list_all = tl.zeros([num_k_rows, head_dim], dtype=tl.float32)

    j = 0
    while j < num_k_rows:
        # k_ptr and v_ptr are flattened so row j is contiguous of length head_dim
        k_list_all[j, :] = tl.load(k_rows_ptr + j * head_dim + tl.arange(0, head_dim))
        v_list_all[j, :] = tl.load(v_rows_ptr + j * head_dim + tl.arange(0, head_dim))
        j += 1

    # Compute logits = q_vec · k_list_all^T -> [num_k_rows]
    logits = tl.zeros([num_k_rows], dtype=tl.float32)
    for j in range(0, num_k_rows):
        dot = 0.0
        for d in range(0, head_dim):
            dot += q_vec[d] * k_list_all[j, d]
        logits[j] = dot

    # Scale logits
    scaled = logits * sm_scale

    # Compute lse = logsumexp(scaled) / ln(2)
    max_scaled = -float('inf')
    for j in range(0, num_k_rows):
        if scaled[j] > max_scaled:
            max_scaled = scaled[j]
    sumexp = 0.0
    for j in range(0, num_k_rows):
        sumexp += tl.exp(scaled[j] - max_scaled)
    lse_val = (max_scaled + tl.log(sumexp)) * 1.44269504  # 1/ln(2)

    # Store lse to lse_ptr
    tl.store(lse_ptr, lse_val)

    # Compute attn = softmax(scaled)
    sumexp_shift = 0.0
    for j in range(0, num_k_rows):
        sumexp_shift += tl.exp(scaled[j] - lse_val)
    attn = tl.zeros([num_k_rows], dtype=tl.float32)
    for j in range(0, num_k_rows):
        attn[j] = tl.exp(scaled[j] - lse_val) / sumexp_shift

    # Compute out_vec = attn · v_list_all -> [head_dim]
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for d in range(0, head_dim):
        dot_v = 0.0
        for j in range(0, num_k_rows):
            dot_v += attn[j] * v_list_all[j, d]
        out_vec[d] = dot_v

    # Store out_vec
    for d in range(0, head_dim):
        tl.store(out_row_ptr + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move tensors to CUDA for Triton
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
        # Flatten k/v across the size-1 dimension: [num_pages, num_kv_heads, head_dim]
        k_cache_f32 = k_cache.squeeze(1).contiguous().to(torch.float32)
        v_cache_f32 = v_cache.squeeze(1).contiguous().to(torch.float32)

        total_q = q_f32.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs and lse
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device='cuda')
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device='cuda')

        # Process each batch interval
        for b in range(len_indptr - 1):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = qo_end - qo_start
            num_kv_tokens = kv_end - kv_start

            # If intervals are invalid, skip
            if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
                continue

            # For each query token in this batch
            for q_idx in range(num_q_tokens):
                global_q_idx = q_idx + qo_start

                # Compute delta for causal consideration
                delta = num_kv_tokens - num_q_tokens
                max_kv_idx = q_idx + 1 + delta
                if max_kv_idx > num_kv_tokens:
                    max_kv_idx = num_kv_tokens

                # If no valid KV entries for this query, skip
                if max_kv_idx <= 0:
                    continue

                # For each query head
                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)  # 32 // 8 = 4

                    # Gather k_ids and v_ids: kv_indices for this batch b
                    # k_ids = kv_indices[kv_start : kv_start + max_kv_idx]
                    k_ids = kv_indices[kv_start: kv_start + max_kv_idx].to(torch.int32).to('cuda')
                    # Flattened row indices in k_ptr/v_ptr: row = k_id * num_kv_heads + kv_head
                    rows_flat = (k_ids * num_kv_heads + kv_head).to(torch.int32).to('cuda')

                    # Gather corresponding rows from k_cache_f32 and v_cache_f32
                    k_rows = k_cache_f32[rows_flat]  # [max_kv_idx, head_dim], float32
                    v_rows = v_cache_f32[rows_flat]  # [max_kv_idx, head_dim], float32

                    # Load q row for this head: q[global_q_idx, h, :]
                    q_row = q_f32[global_q_idx, h, :]  # [head_dim], float32

                    # Output row pointer
                    out_row = output[global_q_idx, h, :]  # [head_dim], float32

                    # Launch Triton kernel for this (b, q_idx, h)
                    attention_bh_kernel[(1,)](
                        q_row, k_rows, v_rows,
                        max_kv_idx, head_dim, float(sm_scale),
                        out_row, lse[global_q_idx, h],
                    )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
