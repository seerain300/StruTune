import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_h_kernel(
    q_vec_ptr,     # *fp32, [D] 1D tensor containing the query vector for (global_q_idx, h)
    output_ptr,    # *bf16, flattened [T, H, D], we will write to a specific offset
    max_active,    # int32, number of valid KV rows for this (b, q_idx, h)
    sm_scale,      # fp32, scaling factor
    D,             # int32, head_dim (e.g., 128)
    k_ptr,         # *fp32, [K, D] contiguous K rows for this (b, h) (we will slice to [max_active, D] before launch)
    v_ptr,         # *fp32, [K, D] contiguous V rows for this (b, h) (we will slice to [max_active, D] before launch)
    OUTPUT_OFFSET, # int32, offset in output_ptr for this (global_q_idx, h): equals (global_q_idx * H + h) * D
    BLOCK_K: tl.constexpr,
):
    # Load query vector
    q_vec = tl.load(q_vec_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Compute logits_scaled = q_vec · k_row for k in 0..BLOCK_K-1 (masked by max_active)
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        # If k >= max_active, set logits to -inf so it doesn't contribute
        if k >= max_active:
            logits_scaled[k] = -float("inf")
        else:
            k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
            prod = q_vec * k_row
            logits_scaled[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Softmax over logits_scaled (masked entries were set to -inf, so they don't contribute)
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
        if k < max_active:
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
        num_segments = len_indptr - 1  # b ranges from 0 to num_segments-1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]

        # Allocate output (bfloat16 as in original)
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
                # GQA mapping: each Q head h uses KV head kv_head = h // 4
                kv_head = h // (num_qo_heads // 8)  # num_kv_heads = 8

                # Gather indices for this segment
                indices = kv_indices[kv_start:kv_end].to(torch.int32)  # length num_kv_tokens

                # k_cache_flat [N, 8, D]; compute row offsets: offset = index * (8 * D) + kv_head * D
                row_offsets = (indices * (8 * head_dim) + kv_head * head_dim).to(torch.int32)
                # Gather rows from k_cache_flat and v_cache_flat
                k_rows = k_cache_flat[row_offsets]  # [num_kv_tokens, D], float32
                v_rows = v_cache_flat[row_offsets]  # [num_kv_tokens, D], float32

                # Store precomputed segments for this (b, h). We'll slice per q_idx launch to max_active.
                k_segments_list[h].append(k_rows)  # each element is [K, D]
                v_segments_list[h].append(v_rows)  # each element is [K, D]

        # Launch Triton kernel for each (b, q_idx, h) triple: compute output and store bfloat16
        for b in range(num_segments):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            # Iterate query tokens
            for q_idx in range(num_q_tokens):
                global_q_idx = qo_start + q_idx

                # Iterate query heads
                for h in range(num_qo_heads):
                    # Reconstruct k_ptr and v_ptr for this (b, h)
                    k_list = k_segments_list[h]  # list of 1 element per b
                    v_list = v_segments_list[h]  # list of 1 element per b
                    assert len(k_list) == 1 and len(v_list) == 1, "Unexpected precompute size"
                    k_full = k_list[0].to(torch.float32).contiguous()  # [K, D]
                    v_full = v_list[0].to(torch.float32).contiguous()  # [K, D]

                    # Determine max_active (implement causal masking)
                    delta = num_kv_tokens - num_q_tokens
                    max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                    # Slice to active rows; Triton kernel expects [K, D], but we pass a max_active-row view
                    # Note: Triton will use max_active to mask; we can pass the full tensors and ignore extra rows.
                    # To be safe, we set BLOCK_K = head_dim and rely on masking.
                    BLOCK_K = head_dim
                    max_active = max_kv_idx

                    # Prepare q_vec: load q[global_q_idx, h] as 1D tensor
                    q_vec = q_f32[global_q_idx, h].contiguous()  # [D] fp32

                    # Compute OUTPUT_OFFSET in flattened output [T, H, D]
                    OUTPUT_OFFSET = (global_q_idx * num_qo_heads + h) * head_dim

                    # Launch Triton kernel for this triple
                    attention_single_q_h_kernel[(1,)](
                        q_vec,                           # q_vec_ptr: [D] fp32
                        output,                          # output_ptr: [T*H*D] bfloat16
                        max_active,                      # int32
                        float(sm_scale),                 # fp32
                        head_dim,                        # int32
                        k_full,                          # k_ptr: [K, D] fp32 (we slice in kernel via max_active)
                        v_full,                          # v_ptr: [K, D] fp32 (we slice in kernel via max_active)
                        OUTPUT_OFFSET,                   # int32
                        BLOCK_K=BLOCK_K,                 # constexpr
                    )

        return output


def run(*args):
    return ModelNew()(*args)
