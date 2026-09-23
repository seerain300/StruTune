import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_ptr,          # *fp32, shape [T, H, D], contiguous
    output_ptr,     # *bf16, shape [T, H, D], contiguous
    lse_ptr,        # *fp32, shape [T, H], contiguous
    k_ptr,          # *fp32, 1D contiguous, length BLOCK_K * D
    v_ptr,          # *fp32, 1D contiguous, length BLOCK_K * D
    sm_scale,       # fp32 scalar
    total_q: tl.constexpr,   # int
    H: tl.constexpr,         # int
    D: tl.constexpr,         # int
    num_q_tokens: tl.constexpr,  # int
    max_kv_idx: tl.constexpr,     # int
    BLOCK_K: tl.constexpr,        # int (e.g., 128)
):
    # Triton programs map to (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Compute global_q_idx (segmented qo_indptr is not needed here since grid encodes segments)
    global_q_idx = b * num_q_tokens + q_idx

    # Load q vector for this (global_q_idx, h): q_ptr is [T, H, D] contiguous
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32, shape [D]

    # Compute logits_scaled: [BLOCK_K]
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # For each k in [0..BLOCK_K-1], load k_row from k_ptr and compute q·k_row
    for k in range(BLOCK_K):
        # Load row k from k_ptr: k_ptr is [BLOCK_K, D] flattened, so row offset = k * D
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32, [D]
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)  # scalar

    # Apply scaling
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

    # Softmax of logits_scaled (we don't need actual softmax_vals because out_vec is computed directly via v_ptr)
    # However, we keep the structure for clarity; since out_vec = sum_k softmax[k] * v_rows[k, :], we can skip explicit softmax and use
    # attn vector implicitly by iterating and summing. Here, we recompute out_vec directly.

    # Compute output vector: out_vec = sum_k (softmax[k] * v_rows[k, :])
    out_vec = tl.zeros((D,), dtype=tl.float32)
    # Softmax using the same m and sum_exp
    for i in range(BLOCK_K):
        out_vec += (tl.exp(logits_scaled[i] - m) / sum_exp) * tl.load(v_ptr + i * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)

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

        # Promote q to fp32 for compute
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # k_cache, v_cache are [num_pages, 1, 8, 128]; squeeze dim=1 => [num_pages, 8, 128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache_f32.squeeze(1)  # [num_pages, 8, 128]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch the Triton kernel for each (b, q_idx, h)
        BLOCK_K = 128  # must be literal constexpr in kernel signature

        for b in range(num_segments):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start

            for q_idx in range(num_q_tokens):
                global_q_idx = qo_start + q_idx

                for h in range(num_qo_heads):
                    gqa_ratio = num_qo_heads // 8
                    kv_head = h // gqa_ratio

                    # Build k_ptr and v_ptr: 1D arrays of length BLOCK_K * D, fill only first num_kv_tokens rows
                    k_rows = torch.empty((BLOCK_K, head_dim), dtype=torch.float32, device=device)
                    v_rows = torch.empty((BLOCK_K, head_dim), dtype=torch.float32, device=device)

                    # For indices = kv_indices[kv_start + 0..num_kv_tokens-1]
                    indices = kv_indices[kv_start:kv_end]  # [num_kv_tokens]
                    N, M, D2 = k_cache_flat.shape
                    assert D2 == head_dim, "head_dim mismatch"

                    # Copy rows into k_rows[v,:] and v_rows[v,:]
                    for kk in range(num_kv_tokens):
                        idx = int(indices[kk].item())
                        row_offset = idx * (M * D2) + kv_head * D2
                        k_rows[kk, :] = k_cache_flat[row_offset:row_offset + D2]
                        v_rows[kk, :] = v_cache_flat[row_offset:row_offset + D2]

                    # Launch kernel with 3D grid: (num_segments, num_q_tokens, num_qo_heads)
                    attention_single_q_idx_h_kernel[(num_segments, num_q_tokens, num_qo_heads)](
                        q_f32, output, lse,
                        k_rows.reshape(BLOCK_K * head_dim).contiguous(),
                        v_rows.reshape(BLOCK_K * head_dim).contiguous(),
                        sm_scale,
                        total_q=total_q,
                        H=num_qo_heads,
                        D=head_dim,
                        num_q_tokens=num_q_tokens,
                        max_kv_idx=min(q_idx + 1, num_kv_tokens),  # causal-like mask
                        BLOCK_K=BLOCK_K,  # constexpr
                    )

        return output, lse


def run(*args):
    return ModelNew()(*args)
