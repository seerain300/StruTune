import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_logits_ptr,  # [len_indptr, total_q, 32, out_len] float32
    lse_ptr,             # [len_indptr, total_q, 32] float32
    total_q,             # int: total number of query tokens across all batches
    sm_scale,            # float32 scalar
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
    out_len = num_kv_tokens * GQA_RATIO  # 4 * num_kv_tokens

    # Base pointer for q vector [qo_start + q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Out buffer base for this (b, q_token, qo_head)
    base_out = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Accumulator for LSE
    sum_exp = 0.0

    # Compute logits vector for all expanded positions
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r
            # Causal condition: kv_pos < (q_token + 1 + delta), delta = num_kv_tokens - num_q_tokens
            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                # Load q row: [HEAD_DIM]
                q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM))
                # Load k row: [HEAD_DIM] for head j at kv_start + j
                k_row_base = k_ptr + (kv_start + j) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
                k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM))
                # Dot product
                val = 0.0
                for i in range(HEAD_DIM):
                    val += q_row[i] * k_row[i]
                val = val * sm_scale
                # Store logits at position kv_pos
                tl.store(output_logits_ptr + base_out + kv_pos, val)
                # Accumulate for LSE
                sum_exp += tl.exp(val)

    # Compute LSE in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    # Store lse for this (b, q_token, qo_head)
    tl.store(lse_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head), lse_val)


@triton.jit
def _compute_output_kernel(
    output_logits_ptr,  # [len_indptr, total_q, 32, out_len] float32
    lse_ptr,             # [len_indptr, total_q, 32] float32
    v_ptr,               # [B, 8, 128] float32
    output_ptr,          # [total_q, 32, 128] float32
    total_q,             # int: total number of query tokens across all batches
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
    out_len = num_kv_tokens * GQA_RATIO

    # Base for output vector for this (b, q_token, qo_head)
    out_base = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * HEAD_DIM

    # Preload lse
    lse_val = tl.load(lse_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head))

    # Accumulate output vector across head_dim
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r
            # Same causal condition
            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                val = tl.load(output_logits_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len + kv_pos)
                attn = tl.exp(val - lse_val)  # softmax across positions

                # Load v[j, r, :] row
                v_row_base = v_ptr + (kv_start + j) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
                v_row = tl.load(v_row_base + tl.arange(0, HEAD_DIM))

                out_vec += attn * v_row

    # Store output vector
    tl.store(output_ptr + out_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be CUDA tensors."
        device = q.device

        # Compute total_q and total_kv based on indptr (assertions in original code)
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = 4

        # Cast to float32 for computation
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Output logits buffer: per batch, we need out_len = num_kv_tokens * 4. Allocate maximum across batches.
        max_out_len = total_kv * gqa_ratio
        output_logits = torch.empty((qo_indptr.numel(), total_q, num_qo_heads, max_out_len), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute logits and lse
        grid1 = (qo_indptr.numel(), total_q, num_qo_heads)
        _compute_logits_and_lse_kernel[grid1](
            q_f32, k_f32, qo_indptr, kv_indptr,
            output_logits, lse,
            total_q,
            sm_scale,
            NUM_QO_HEADS=num_qo_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, GQA_RATIO=gqa_ratio,
        )

        # Launch Triton kernel to compute output
        grid2 = (qo_indptr.numel(), total_q, num_qo_heads)
        _compute_output_kernel[grid2](
            output_logits, lse, v_f32, output,
            total_q,
            NUM_QO_HEADS=num_qo_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, GQA_RATIO=gqa_ratio,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
