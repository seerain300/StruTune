import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_lse_kernel(
    q_ptr, k_ptr, qo_indptr_ptr, kv_indptr_ptr,
    lse_ptr,          # [len_indptr, total_q, 32] float32
    sm_scale,         # float32
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch ranges
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start

    # Base pointer for q vector: q[b, q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # LSE accumulator
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time)
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r  # expanded position index
            # Causal mask: allow if kv_pos < q_token + 1 + delta, where delta = num_kv_tokens - num_q_tokens
            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                # Compute dot: q_vec dot k_vec
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM)  # head dim 128
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                prod = q_row * k_row
                dot = tl.sum(prod, axis=0)
                val = dot * sm_scale
                # Accumulate for LSE
                sum_exp += tl.exp(val)
            # else: masked, contribution 0

    # Compute logsumexp in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse for this (b, q_token, qo_head)
    lse_index = b * (qo_end * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


@triton.jit
def _compute_output_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    lse_ptr, output_ptr,     # [total_q, 32, 128] float32
    sm_scale,                # float32
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch ranges
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start

    # Base pointer for q vector: q[b, q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Output base for this (q_token, qo_head)
    out_row_base = output_ptr + (q_token * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM)

    # LSE for this (b, q_token, qo_head)
    lse_index = b * (qo_end * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    lse_val = tl.load(lse_ptr + lse_index)

    # Initialize output vector for this head
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Loop over original KV heads and GQA expansions (compile-time)
    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r
            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                # Compute attention for this position: exp((val - lse) * sm_scale)
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM)
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                prod = q_row * k_row
                dot = tl.sum(prod, axis=0)
                val = dot * sm_scale
                attn = tl.exp((val - lse_val) * sm_scale)
                # Load V for this original KV head j across 128 dims
                v_row_base = v_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM)
                v_row = tl.load(v_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                out_vec += attn * v_row

    # Store output for this (q_token, qo_head)
    tl.store(out_row_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure tensors are on CUDA device and contiguous; keep dtype float32 for stability
        device = q.device
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)

        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Allocate lse buffer
        lse = torch.empty(
            (len_indptr, total_q, 32),
            dtype=torch.float32,
            device=device
        )

        # Launch kernel 1: compute lse per (b, q_token, qo_head)
        grid = (len_indptr, total_q, 32)
        _compute_lse_kernel[grid](
            q, k, qo_indptr, kv_indptr,
            lse,
            sm_scale,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
            num_warps=4, num_stages=2
        )

        # Allocate output
        output = torch.empty(
            (total_q, 32, 128),
            dtype=torch.float32,
            device=device
        )

        # Launch kernel 2: compute output per (b, q_token, qo_head) using lse
        _compute_output_kernel[grid](
            q, k, v, qo_indptr, kv_indptr,
            lse, output,
            sm_scale,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
