import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel_perhead(
    q_ptr,           # *fp32, shape [T, H, D], contiguous
    k_ptr_flat,      # *fp32, shape [num_kv_tokens * D], contiguous (per-head packed)
    v_ptr_flat,      # *fp32, shape [num_kv_tokens * D], contiguous (per-head packed)
    output_ptr,      # *bf16, shape [T, H, D], contiguous
    lse_ptr,         # *fp32, shape [T, H], contiguous (atomic add)
    sm_scale,        # fp32 scalar
    total_q,         # int32
    H: tl.constexpr, # num_qo_heads (e.g., 32)
    D: tl.constexpr, # head_dim (e.g., 128)
    num_q_tokens,    # int32
    num_kv_tokens,   # int32 (max_kv_idx)
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    # Program ids: dummy for segments, q_idx over queries, h over heads
    b = tl.program_id(0)  # we don't use b here; for single-segment, this is 0
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    global_q_idx = b * num_q_tokens + q_idx

    # Load q vector for this (global_q_idx, h): q_ptr is [T, H, D] contiguous
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32 [D]

    # Compute logits_scaled: [BLOCK_K], initialize to -inf (we'll fill valid ones)
    logits_scaled = tl.full((BLOCK_K,), -float('inf'), dtype=tl.float32)

    # For each k in [0..BLOCK_K-1], compute dot(q_vec, k_row) and store
    for k in range(BLOCK_K):
        valid = k < num_kv_tokens
        # Load per-head k_row at offset k * D (vector of length D)
        k_row = tl.load(k_ptr_flat + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32 [D]
        prod = q_vec * k_row
        dot = tl.sum(prod, axis=0)  # scalar
        logits_scaled[k] = dot

    # Scale
    logits_scaled = logits_scaled * sm_scale

    # Logsumexp in base-2
    m = tl.max(logits_scaled, axis=0)
    sum_exp = tl.sum(tl.exp(logits_scaled - m), axis=0)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)

    # Atomic add lse for (global_q_idx, h)
    tl.atomic_add(lse_ptr + global_q_idx * H + h, lse_base2)

    # Compute output vector: out_vec = sum_k (softmax[k] * v_rows[k, :]) using per-head packed v_ptr_flat
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(BLOCK_K):
        attn_i = tl.exp(logits_scaled[i] - m) / sum_exp
        v_row = tl.load(v_ptr_flat + i * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32 [D]
        out_vec += attn_i * v_row

    # Store output as bf16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        #


def run(*args):
    return ModelNew()(*args)
