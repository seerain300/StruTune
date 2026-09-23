import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, v_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    logits_ptr,           # [len_indptr, total_q, 32, 32] float32
    lse_ptr,               # [len_indptr, total_q, 32] float32
    total_q,               # int
    q_token_plus1,         # int: q_token + 1 + delta (passed from host)
    sm_scale,              # float32
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
    out_len = NUM_KV_HEADS * GQA_RATIO  # 32

    # Base pointer for q vector: q[b, q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Base index for logits buffer for this (b, q_token, qo_head)
    base_out = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Accumulator for LSE
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time loops)
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r  # expanded position index

            # Causal mask: allow if kv_pos < q_token_plus1, where q_token_plus1 = q_token + 1 + delta
            if kv_pos < q_token_plus1:
                # Compute dot: q_vec dot k_vec
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM)  # kv_pos selects original KV head
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                prod = q_row * k_row
                dot = tl.sum(prod, axis=0)
                val = dot * sm_scale
                # Store logits
                tl.store(logits_ptr + base_out + kv_pos, val)
                # Accumulate for LSE
                sum_exp += tl.exp(val)
            else:
                tl.store(logits_ptr + base_out + kv_pos, -float("inf"))

    # Compute logsumexp in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse for this (b, q_token, qo_head)
    lse_index = b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


@triton.jit
def _compute_output_kernel(
    logits_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_ptr,           # [total_q, 32, 128] float32
    lse_ptr,               # [len_indptr, total_q, 32] float32
    total_q,               # int
    q_token_plus1,         # int: q_token + 1 + delta
    sm_scale,              # float32
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
    out_len = NUM_KV_HEADS * GQA_RATIO  # 32

    # Base pointer for q vector: q[b, q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Base index for logits buffer for this (b, q_token, qo_head)
    base_out = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Initialize output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Loop over original KV heads and GQA expansions (compile-time)
    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r
            # Read logits and subtract lse
            lse_index = b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
            lse_val = tl.load(lse_ptr + lse_index)
            val = tl.load(logits_ptr + base_out + kv_pos) - lse_val
            attn = tl.exp(val)
            # Accumulate output: out_vec += attn * V[j, :]
            v_row_base = v_ptr + j * (NUM_KV_HEADS * HEAD_DIM)
            v_row = tl.load(v_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
            out_vec += attn * v_row

    # Store final output vector
    out_index = (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM
    tl.store(output_ptr + out_index + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # Cast to float32 for numeric stability
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        total_q = int(q_f32.shape[0])
        num_qo_heads = 32
        head_dim = 128
        total_kv = int(k_f32.shape[0])
        num_kv_heads = 8
        gqa_ratio = 4  # fixed

        # Allocate logits buffer: [len_indptr, total_q, 32, 32] float32
        len_indptr = qo_indptr.shape[0]
        out_len = num_kv_heads * gqa_ratio  # 32
        logits = torch.empty((len_indptr, total_q, num_qo_heads, out_len), dtype=torch.float32, device=q.device)
        lse = torch.empty((len_indptr, total_q, num_qo_heads), dtype=torch.float32, device=q.device)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)

        # Compute q_token_plus1 = q_token + 1 + delta per batch and pass to kernels
        q_token_plus1 = []
        for b in range(len_indptr):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start
            num_kv_tokens = kv_end - kv_start
            delta = num_kv_tokens - num_q_tokens
            q_token_plus1.append(num_q_tokens + 1 + delta)
        q_token_plus1 = torch.tensor(q_token_plus1, dtype=torch.int32, device=q.device)

        # Launch Triton kernel 1: compute logits and LSE
        grid1 = (len_indptr, total_q, num_qo_heads)
        _compute_logits_and_lse_kernel[grid1](
            q_f32, k_f32, v_f32,
            qo_indptr, kv_indptr,
            logits, lse,
            total_q, q_token_plus1, float(sm_scale),
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
            num_warps=4, num_stages=2
        )

        # Launch Triton kernel 2: compute output
        grid2 = (len_indptr, total_q, num_qo_heads)
        _compute_output_kernel[grid2](
            logits, v_f32, qo_indptr, kv_indptr,
            output, lse,
            total_q, q_token_plus1, float(sm_scale),
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
            num_warps=4, num_stages=2
        )

        # Return output and lse. lse shape matches original: [len_indptr, total_q, 32]
        return output, lse


def run(*args):
    return ModelNew()(*args)
