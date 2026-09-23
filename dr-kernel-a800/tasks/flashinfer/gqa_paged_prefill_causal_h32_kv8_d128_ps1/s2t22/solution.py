import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_vec_ptr,     # *fp32, 1D [D] query vector for (global_q_idx, h)
    output_ptr,    # *bf16, flattened [T, H, D] output
    k_ptr,         # *fp32, [K, D] K rows of k for this triple
    v_ptr,         # *fp32, [K, D] K rows of v for this triple
    K,             # int32, number of active KV rows (max_active)
    sm_scale,      # fp32, scaling factor
    D,             # int32, head_dim (e.g., 128)
    OUTPUT_OFFSET, # int32, offset in output_ptr for (global_q_idx, h): equals (global_q_idx * H + h) * D
    BLOCK_K: tl.constexpr,  # compile-time loop bound, should be >= K (e.g., 128)
):
    # Load query vector (1D)
    q_vec = tl.load(q_vec_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Compute logits_scaled = q_vec · k_row for k in 0..BLOCK_K-1; mask k >= K by setting -inf
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        if k >= K:
            logits_scaled[k] = -float("inf")
        else:
            k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
            prod = q_vec * k_row
            logits_scaled[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Softmax
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
        if k < K:
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
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        # k_cache and v_cache are [N, 1, 8, D]; squeeze dim=1 => [N, 8, D]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()  # [len_indptr]
        kv_indptr = kv_indptr.to(torch.int32).contiguous()  # [len_indptr]
        kv_indices = kv_indices.to(torch.int32).contiguous()  # [num_kv_indices]


def run(*args):
    return ModelNew()(*args)
