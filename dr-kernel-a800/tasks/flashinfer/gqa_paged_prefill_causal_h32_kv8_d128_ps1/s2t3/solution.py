import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_ptr,          # *fp32, [T, H, D], flattened
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    kv_indices_ptr, # *int32, [num_kv_indices]
    output_ptr,     # *bf16, [T, H, D], flattened
    lse_ptr,        # *fp32, [T, H]
    sm_scale,       # fp32
    T,              # int32, total_q
    H,              # int32, num_qo_heads
    D,              # int32, head_dim
    BLOCK_K: tl.constexpr,
):
    # Grid: (b in 0..len_indptr-2, q_idx in 0..num_q_tokens-1, h in 0..H-1)
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

    # GQA mapping: 8 KV heads shared across 32 Q heads
    gqa_ratio = H // 8  # since num_qo_heads=32, num_kv_heads=8
    kv_head = h // gqa_ratio

    # Compute logits = q_vec @ k_rows for k in 0..BLOCK_K-1, mask k < max_kv_idx
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        kv_index_k = tl.load(kv_indices_ptr + kv_start + k, mask=valid, other=0)
        # k_cache/v_cache are originally [N,1,8,128]; squeezing dim=1 => [N,8,128]
        # Row offset: row_index = kv_index_k * (8 * D) + kv_head * D
        offset = kv_index_k * (8 * D) + kv_head * D
        k_row = tl.load(q_ptr + offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        prod = q_vec * k_row
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

    # Compute softmax of logits_scaled (masked k >= max_kv_idx contribute 0)
    for i in range(BLOCK_K):
        logits_scaled[i] = -float('inf') if (i >= max_kv_idx) else logits_scaled[i]
    denom = 0.0
    for i in range(BLOCK_K):
        denom += tl.exp(logits_scaled[i] - m)
    softmax_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(BLOCK_K):
        softmax_vals[i] = tl.exp(logits_scaled[i] - m) / denom

    # Compute output vector: out_vec += softmax[k] * v_rows[k, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        kv_index_k = tl.load(kv_indices_ptr + kv_start + k, mask=valid, other=0)
        offset = kv_index_k * (8 * D) + kv_head * D
        v_row = tl.load(q_ptr + offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        out_vec += softmax_vals[k] * v_row

    # Store output as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        # k_cache and v_cache are [N,1,8,128]; squeeze dim=1 => [N,8,128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Number of segments = len_indptr - 1
        num_segments = len_indptr - 1

        # Launch Triton kernel over 3D grid: (segments, T, H)
        grid = (num_segments, total_q, num_qo_heads)

        # Set BLOCK_K to head_dim=128; mask k >= max_kv_idx to handle varying K
        attention_single_q_idx_h_kernel[grid](
            q_f32, qo_indptr, kv_indptr, kv_indices, output, lse, sm_scale,
            total_q, num_qo_heads, head_dim,
            BLOCK_K=128,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
