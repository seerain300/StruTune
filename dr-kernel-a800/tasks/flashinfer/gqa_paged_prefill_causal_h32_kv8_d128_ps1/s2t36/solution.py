import math
import torch
import triton
import triton.language as tl


@triton.jit
def gqa_kernel(
    q_ptr,              # *fp32, [T, H, D] contiguous
    k_ptr,              # *fp32, [MAX_K * D] contiguous (selected rows for this segment)
    v_ptr,              # *fp32, [MAX_K * D] contiguous
    output_ptr,         # *bf16, [T, H, D] contiguous
    lse_ptr,            # *fp32, [T, H] contiguous
    sm_scale,           # fp32 scalar
    total_q: tl.constexpr,   # int, not used directly; for completeness
    H: tl.constexpr,         # 32
    D: tl.constexpr,         # 128
    max_kv_idx: tl.constexpr,  # int per (segment, q_idx, h), e.g., 34
    MAX_K: tl.constexpr,       # 128 (compile-time upper bound)
):
    # Program ids map to (segment b, q_idx within segment, head h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # global_q_idx = q_idx since len_indptr=2 and we assume single segment
    global_q_idx = q_idx

    # Load q vector for this head as fp32, shape [D]
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], fp32

    # Compute logits_scaled for k in [0..MAX_K-1]
    logits_scaled = tl.zeros((MAX_K,), dtype=tl.float32)

    for k in range(MAX_K):
        # Load k_row as fp32 vector [D]
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], fp32
        prod = q_vec * k_row
        sum_prod = 0.0
        for j in range(D):
            sum_prod += prod[j]
        logits_scaled[k] = sum_prod  # scalar

    # Apply scaling
    logits_scaled = logits_scaled * sm_scale

    # Mask out entries beyond max_kv_idx by setting them to -inf so they don't affect logsumexp
    for k in range(MAX_K):
        if k >= max_kv_idx:
            logits_scaled[k] = -float("inf")

    # Compute logsumexp (natural log), then convert to base-2
    m = logits_scaled[0]
    for k in range(1, MAX_K):
        m = tl.maximum(m, logits_scaled[k])
    sum_exp = 0.0
    for k in range(MAX_K):
        sum_exp += tl.exp(logits_scaled[k] - m)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)

    # Store lse for this (global_q_idx, h) as fp32
    lse_offset = global_q_idx * H + h
    tl.store(lse_ptr + lse_offset, lse_base2)

    # Compute output vector: out_vec = sum_k (softmax_k * v_row[k, :])
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for k in range(MAX_K):
        attn_k = tl.exp(logits_scaled[k] - m) / sum_exp  # scalar
        v_row = tl.load(v_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], fp32
        out_vec += attn_k * v_row

    # Store output vector for (global_q_idx, h) as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    out_vec_bf = out_vec.to(tl.bfloat16)
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec_bf, mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        # k_cache and v_cache are [N, 1, 8, 128]; squeeze dim=1 => [N, 8, 128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q,


def run(*args):
    return ModelNew()(*args)
