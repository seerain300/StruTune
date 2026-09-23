import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    logits_ptr,           # [len_indptr, total_q, 32, 32] float32
    lse_ptr,              # [len_indptr, total_q, 32] float32
    sm_scale,             # float32 scalar
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
    out_len = NUM_KV_HEADS * GQA_RATIO  # 32 for given setup

    # Base pointers for q vector
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Output logits base
    base_out = (b * (num_q_tokens * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Accumulator for LSE
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time loops)
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r  # expanded position index

            # Causal mask: can only attend to positions < q_token + 1 + (num_kv_tokens - num_q_tokens)
            delta = num_kv_tokens - num_q_tokens
            if kv_pos < (q_token + 1 + delta):
                # Compute dot: q_vec dot k_vec
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
                k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM)  # head dim 128
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
                # masked positions contribute 0 to sum_exp

    # Compute logsumexp in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse for this (b, q_token, qo_head)
    lse_index = b * (num_q_tokens * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


@triton.jit
def _compute_output_kernel(
    logits_ptr, v_ptr, kv_indptr_ptr, qo_indptr_ptr,
    output_ptr,           # [total_q, 32, 128] float32
    lse_ptr,              # [len_indptr, total_q, 32] float32
    sm_scale,             # float32 scalar
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

    # LSE for this (b, q_token, qo_head)
    lse_index = b * (num_q_tokens * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head
    lse_val = tl.load(lse_ptr + lse_index)

    # Base pointers
    base_out = (b * (num_q_tokens * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * HEAD_DIM
    output_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Initialize output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Accumulate output across allowed positions
    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r
            # If masked, contribution is zero
            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                val = tl.load(logits_ptr + (b * (num_q_tokens * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len + kv_pos)
                attn = tl.exp(val - lse_val)
                # v_expanded row: original v at kv_pos (j*4 + r) and expanded along head_dim
                v_row_base = v_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + r * HEAD_DIM
                v_row = tl.load(v_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)
                out_vec += attn * v_row
            # else: masked, do nothing

    # Store final output vector
    tl.store(output_ptr + base_out, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA"
        device = q.device
        # Cast to float32 for computation
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = 4

        len_indptr = qo_indptr.shape[0]

        # Allocate buffers
        logits = torch.empty(
            (len_indptr, total_q, num_qo_heads, num_kv_heads * gqa_ratio),
            dtype=torch.float32,
            device=device,
        )
        lse = torch.empty(
            (len_indptr, total_q, num_qo_heads),
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            (total_q, num_qo_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )

        # Launch Triton kernel 1: compute logits and LSE
        grid = (len_indptr, total_q, num_qo_heads)
        _compute_logits_and_lse_kernel[grid](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            logits, lse,
            sm_scale,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
        )

        # Launch Triton kernel 2: compute output from logits and v
        _compute_output_kernel[grid](
            logits, v_f32, kv_indptr, qo_indptr,
            output, lse,
            sm_scale,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
        )

        # Return (output, lse)
        return output, lse


def run(*args):
    return ModelNew()(*args)
