import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_h_kernel(
    q_vec_ptr,     # *fp32, [D], 1D tensor of the query vector for (global_q_idx, h)
    output_ptr,    # *bf16, flattened [T, H, D], we will write to a specific offset
    max_active,    # int32, number of active KV rows for this (b, q_idx, h)
    sm_scale,      # fp32
    D,             # int32, head_dim
    k_ptr,         # *fp32, [max_active, D], contiguous
    v_ptr,         # *fp32, [max_active, D], contiguous
    OUTPUT_OFFSET, # int32, offset in output_ptr for this (global_q_idx, h): equals global_q_idx*H*D + h*D
    BLOCK_K: tl.constexpr,
):
    # Load q vector for this head: q_vec_ptr is 1D [D]
    q_vec = tl.load(q_vec_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Compute logits_scaled = q_vec · k_rows for k in 0..BLOCK_K-1
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Softmax over logits_scaled
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
        v_row = tl.load(v_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        out_vec += softmax_vals[k] * v_row

    # Store output as bfloat16 to output_ptr at [OUTPUT_OFFSET :]
    tl.store(output_ptr + OUTPUT_OFFSET + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        # k_cache and v_cache are [N, 1, 8, D]; squeeze dim=1 => [N, 8, D]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)

        # Precompute k_segments and v_segments per segment b for all heads h
        k_segments_list = [[] for _ in range(num_qo_heads)]
        v_segments_list = [[] for _ in range(num_qo_heads)]

        # For each segment b
        for b in range(num_segments):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start

            # Build k_segments[h] and v_segments[h] for this segment
            for h in range(num_qo_heads):
                gqa_ratio = num_qo_heads // 8  # num_kv_heads =


def run(*args):
    return ModelNew()(*args)
