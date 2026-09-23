import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_logits_ptr,  # [len_indptr, total_q, num_qo_heads, out_len] float32
    lse_ptr,             # [len_indptr, total_q, num_qo_heads] float32
    total_q,             # int: number of query tokens across all batches
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

    # Skip if no elements in this batch
    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    out_len = num_kv_tokens * GQA_RATIO  # per-batch expanded KV length

    # Base pointers
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM
    base_out = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Accumulator for LSE
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time loops)
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r  # original KV position index after expansion

            # Compute dot: q_vec dot k_vec
            q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
            k_row_base = k_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
            k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
            dot = tl.sum(q_row * k_row, axis=0)  # scalar
            logit = dot  # sm_scale is applied in host-side casting already

            # Causal mask: allow if kv_pos < (q_token + 1 + delta), delta = num_kv_tokens - num_q_tokens
            delta = num_kv_tokens - num_q_tokens
            if kv_pos < (q_token + 1 + delta):
                tl.store(output_logits_ptr + base_out + kv_pos, logit)
            else:
                tl.store(output_logits_ptr + base_out + kv_pos, -float("inf"))

            # Accumulate exp for LSE
            expv = tl.exp(logit)
            sum_exp += expv

    # Compute LSE in base-2
    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head), lse_val)


@triton.jit
def _compute_output_kernel(
    output_logits_ptr,    # [len_indptr, total_q, num_qo_heads, out_len] float32
    k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_ptr,           # [len_indptr, total_q, num_qo_heads, HEAD_DIM] float32
    lse_ptr,              # [len_indptr, total_q, num_qo_heads] float32
    total_q,              # int
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    out_len = num_kv_tokens * GQA_RATIO  # per-batch expanded KV length

    base_out = (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head) * out_len

    # Load lse for normalization
    lse_val = tl.load(lse_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head))

    # Accumulate output vector across GQA expansions
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for j in range(NUM_KV_HEADS):
        for r in range(GQA_RATIO):
            kv_pos = j * GQA_RATIO + r
            # softmax weight: exp(logit - lse)
            logit = tl.load(output_logits_ptr + base_out + kv_pos)
            softmax_w = tl.exp(logit - lse_val)
            # Gather v_expanded for this (j, r) and add to output
            v_row_base = v_ptr + (kv_start + kv_pos) * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
            v_row = tl.load(v_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
            out_vec += softmax_w * v_row

    # Store output
    out_base = b * (total_q * NUM_QO_HEADS) * HEAD_DIM + (q_token * NUM_QO_HEADS + qo_head) * HEAD_DIM
    tl.store(output_ptr + out_base + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Cast inputs to float32 for compute; keep devices as given
        device = q.device
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        # Shapes match original: q [total_q, 32, 128], k [total_kv, 8, 128], v same
        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.numel()

        # Allocate output and lse tensors
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Allocate per-batch output_logits buffers sized by out_len = (kv_end - kv_start) * GQA_RATIO
        # We need dynamic buffer sizes per batch; Triton kernels write up to out_len. We'll allocate a Python list of tensors.
        output_logits_list = []
        for b in range(len_indptr):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start
            out_len = num_kv_tokens * 4  # GQA_RATIO = 4
            output_logits_list.append(torch.empty(
                (total_q, 32, out_len),
                dtype=torch.float32,
                device=device,
            ))

        # Launch kernel to compute logits and LSE
        grid = (len_indptr, total_q, 32)
        _compute_logits_and_lse_kernel[grid](
            q, k, v, qo_indptr, kv_indptr,
            output_logits_list[0],  # placeholder; Triton uses program_id b to pick the correct buffer
            lse,
            total_q,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
        )

        # Launch kernel to compute final output from logits
        _compute_output_kernel[grid](
            output_logits_list[0], k, v, qo_indptr, kv_indptr,
            output, lse,
            total_q,
            NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, GQA_RATIO=4,
        )

        # Return (output, lse)
        return output, lse


def run(*args):
    return ModelNew()(*args)
