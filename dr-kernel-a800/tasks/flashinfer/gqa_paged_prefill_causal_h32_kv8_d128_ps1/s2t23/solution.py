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
    K,             # int32, number of active KV rows
    sm_scale,      # fp32, scaling factor
    D,             # int32, head_dim (e.g., 128)
    OUTPUT_OFFSET, # int32, offset in output_ptr for (global_q_idx, h): equals (global_q_idx * H + h) * D
    BLOCK_K: tl.constexpr,  # compile-time loop bound, e.g., 128
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

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1  # b in [0, num_segments)

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]

        # Allocate output (bfloat16 as in original)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)

        # For each segment b
        for b in range(num_segments):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start

            # Iterate query tokens
            for q_idx in range(num_q_tokens):
                global_q_idx = qo_start + q_idx

                # Iterate query heads
                for h in range(num_qo_heads):
                    # GQA mapping: kv_head = h // 4 (since H=32, M=8, ratio=4)
                    kv_head = h // 4

                    # Gather K/V rows for this (b, h): indices = kv_indices[kv_start:kv_end]
                    indices = kv_indices[kv_start:kv_end].to(torch.int32)  # [num_kv_tokens]
                    N, M, D2 = k_cache_flat.shape
                    assert D2 == head_dim, "head_dim mismatch"
                    # Row offsets into k_cache_flat: indices * (8*D) + kv_head * D
                    row_offsets = (indices * (8 * D2) + kv_head * D2).to(torch.int32)
                    k_rows = k_cache_flat[row_offsets]  # [K, D], contiguous
                    v_rows = v_cache_flat[row_offsets]  # [K, D], contiguous

                    # Prepare q_vec: load q[global_q_idx, h] as 1D fp32
                    q_vec = q_f32[global_q_idx, h].contiguous()  # [D] fp32

                    # Compute OUTPUT_OFFSET in flattened output [T, H, D]
                    OUTPUT_OFFSET = (global_q_idx * num_qo_heads + h) * head_dim

                    # Launch Triton kernel for this triple
                    attention_single_q_idx_h_kernel[(1,)](
                        q_vec,                           # q_vec_ptr: [D] fp32
                        output,                          # output_ptr: [T*H*D] bfloat16 (flattened view)
                        k_rows,                          # k_ptr: [K, D] fp32
                        v_rows,                          # v_ptr: [K, D] fp32
                        num_kv_tokens,                   # K
                        float(sm_scale),                 # fp32
                        head_dim,                        # int32
                        OUTPUT_OFFSET,                   # int32
                        BLOCK_K=head_dim,                # constexpr loop bound (e.g., 128)
                    )

        return output


def run(*args):
    return ModelNew()(*args)
