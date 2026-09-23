import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_triton_kernel(
    q_ptr,            # *fp32, [T, H, D], flattened
    k_ptr,            # *fp32, [N, 8, D] (k_cache_flat)
    v_ptr,            # *fp32, [N, 8, D] (v_cache_flat)
    kv_indices_ptr,   # *int32, [num_kv_indices]
    output_ptr,       # *bf16, [T, H, D], flattened
    lse_ptr,          # *fp32, [T, H]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    sm_scale,         # fp32
    T,                # int32, total_q
    H,                # int32, num_qo_heads
    D,                # int32, head_dim
    N,                # int32, num_pages (for k_ptr/v_ptr indexing of k_cache, v_cache)
):
    # Grid: (b in 0..len_indptr-2, q_idx in 0..T-1, h in 0..H-1)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Compute segment starts/ends
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    global_q_idx = qo_start + q_idx
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx = tl.minimum(q_idx + 1 + delta, num_kv_tokens)

    # Load q vector for this head: q[global_q_idx, h]
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Load kv indices for this segment
    kv_indices_seg = kv_indices_ptr[kv_start:kv_end]  # [num_kv_tokens]
    # Compute k rows: k_cache_flat[page_ids, kv_head, :]
    # GQA mapping: kv_head = h // 4
    gqa_ratio = H // 8  # since num_qo_heads == 32 and num_kv_heads == 8
    kv_head = h // gqa_ratio
    # We need to load k_rows and v_rows for k in 0..max_kv_idx-1
    # For simplicity and correctness, we loop k up to BLOCK_K (set on host), masking k >= max_kv_idx
    # Note: k_ptr and v_ptr are [N, 8, D]. Row index = page_id * (8*D) + kv_head * D
    # For each k, index = (kv_indices_seg[k] * (8*D)) + kv_head * D
    BLOCK_K = 128  # head_dim, safe upper bound; mask ensures we only use max_kv_idx
    k_rows = [None] * BLOCK_K
    v_rows = [None] * BLOCK_K
    for k in range(BLOCK_K):
        if k < num_kv_tokens:
            idx = kv_indices_seg[k] * (8 * D) + kv_head * D
            k_rows[k] = tl.load(k_ptr + idx + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
            v_rows[k] = tl.load(v_ptr + idx + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        else:
            k_rows[k] = tl.zeros((D,), dtype=tl.float32)
            v_rows[k] = tl.zeros((D,), dtype=tl.float32)

    # Compute logits = q_vec @ k_rows.T -> [BLOCK_K]
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        prod = q_vec * k_rows[k]
        logits_scaled[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Compute logsumexp in base-2
    m = logits_scaled[0]
    for i in range(1, BLOCK_K):
        m = tl.maximum(m, logits_scaled[i])
    sum_exp = 0.0
    for i in range(BLOCK_K):
        sum_exp += tl.exp(logits_scaled[i] - m)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)
    # Atomic add lse contribution for this (global_q_idx, h)
    tl.atomic_add(lse_ptr + global_q_idx * H + h, lse_base2)

    # Compute softmax of logits_scaled (mask k >= max_kv_idx to -inf)
    for i in range(BLOCK_K):
        logits_scaled[i] = -float('inf') if (i >= max_kv_idx) else logits_scaled[i]
    m = logits_scaled[0]
    for i in range(1, BLOCK_K):
        m = tl.maximum(m, logits_scaled[i])
    denom = 0.0
    for i in range(BLOCK_K):
        denom += tl.exp(logits_scaled[i] - m)
    softmax_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(BLOCK_K):
        softmax_vals[i] = tl.exp(logits_scaled[i] - m) / denom

    # Compute output vector: out_vec += softmax[k] * v_rows[k, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for k in range(BLOCK_K):
        v_row = v_rows[k]
        out_vec += softmax_vals[k] * v_row

    # Store output as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()  # [N, 1, 8, D]
        v_cache_f32 = v_cache.to(torch.float32).contiguous()  # [N, 1, 8, D]
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]
        N, M, D2 = k_cache_flat.shape
        assert D2 == head_dim, "head_dim mismatch between q and k_cache"

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel with a proper 3D grid
        grid = (num_segments, total_q, num_qo_heads)
        attention_single_triton_kernel[grid](
            q_f32, k_cache_flat, v_cache_flat, kv_indices, output, lse, qo_indptr, kv_indptr, sm_scale,
            total_q, num_qo_heads, head_dim, N,
            BLOCK_K=head_dim,  # we only need up to 128 here
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
