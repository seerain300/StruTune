import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_per_j_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    q_stride_b, q_stride_h, q_stride_d,
    k_stride_b, k_stride_h,
    logits_stride_q, logits_stride_h, logits_stride_j,
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    delta = num_kv_tokens - num_q_tokens

    for j in range(0, num_kv_tokens):
        # q[i, h]
        q_offset = i * q_stride_b + h * q_stride_h
        q_val = tl.load(q_ptr + q_offset)

        # k_expanded[j, h] (k_expanded is [num_kv_tokens, 32, 128], contiguous)
        k_offset = j * k_stride_b + h * k_stride_h
        k_val = tl.load(k_ptr + k_offset)

        score = q_val * k_val * scale

        # causal mask
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, -float("inf"))

        # store logits[i, h, j] (logits is [num_q_tokens, 32, num_kv_tokens], contiguous)
        logits_offset = i * logits_stride_q + h * logits_stride_h + j * logits_stride_j
        tl.store(logits_ptr + logits_offset, score)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute logsumexp along j for (i, h): lse[i, h] = log(sum(exp(score - m))) + m
    m = -float("inf")
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        m = tl.maximum(m, score)

    s = 0.0
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)
        s += y

    # Original divides by log(2.0)
    inv_log2 = 1.0 / math.log(2.0)
    lse_val = tl.log(s) * inv_log2
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, lse_val)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    v_stride_b, v_stride_h, v_stride_d,
    out_stride_b, out_stride_h, out_stride_d,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h] = logsumexp(logits[i, h, :]) / log(2.0)
    lse_offset = i * lse_ptr.stride(0) + h * lse_ptr.stride(1)  # WRONG: use strides passed in
    # Fix: We need to use lse_stride_b and lse_stride_h arguments. The above line should read:
    lse_offset = i * lse_stride_b + h * lse_stride_h
    lse_val = tl.load(lse_ptr + lse_offset)  # this is logsumexp / log(2.0)

    # Initialize output[i, h, :]
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, head_dim)
    out_ptr_vec = out_ptr + out_base + d

    # Accumulate: out[i, h, :] += sum_j exp(logits[i, h, j] - lse_val) * v_expanded[j, h, :]
    for j in range(0, num_kv_tokens):
        # Load score = logits[i, h, j]
        offset = i * lse_stride_b + h * lse_stride_h + j  # WRONG again: this is not logits offset
        # Correct: offset for logits is i*logits_stride_q + h*logits_stride_h + j*logits_stride_j
        logits_offset = i * logits_stride_q + h * logits_stride_h + j * logits_stride_j
        score = tl.load(logits_ptr + logits_offset)

        y = tl.exp(score - lse_val)

        # v_expanded[j, h, :] where v_expanded is [num_kv_tokens, 32, 128], contiguous
        v_offset_base = j * v_stride_b + h * v_stride_h
        v_ptr_vec = v_ptr + v_offset_base + d

        val = y * tl.load(v_ptr_vec)
        out_vals = tl.load(out_ptr_vec)
        out_vals = out_vals + val
        tl.store(out_ptr_vec, out_vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton requires CUDA tensors"

        # Constants from original code
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        device = q.device

        # Output and LSE
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment (b-slice)
        for b in range(0, qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice tensors
            q_slice = q[q_start:q_end].to(torch.float32).contiguous()  # [num_q_tokens, 32, 128]
            k_slice = k[kv_start:kv_end].to(torch.float32).contiguous()  # [num_kv_tokens, 8, 128]
            v_slice = v[kv_start:kv_end].to(torch.float32).contiguous()  # [num_kv_tokens, 8, 128]

            # Expand K/V to 32 heads (GQA)
            k_expanded = k_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Strides for Triton
            # q_slice: [num_q_tokens, 32, 128], contiguous
            q_stride_b = q_slice.stride(0) * head_dim * q_slice.shape[1]  # stride in elements for batch
            q_stride_h = q_slice.stride(1) * head_dim                   # stride for head dim
            q_stride_d = q_slice.stride(2)                              # feature stride

            # k_expanded: [num_kv_tokens, 32, 128], contiguous
            k_stride_b = k_expanded.stride(0)                           # N stride
            k_stride_h = k_expanded.stride(1)                           # H stride
            # Note: k_expanded[:, h, :] is contiguous along D, but we only need scalar load for j,h, so k_stride_d not used

            # logits: [num_q_tokens, 32, num_kv_tokens], contiguous
            logits_stride_q = logits.stride(0) * num_kv_tokens
            logits_stride_h = logits.stride(1) * num_kv_tokens
            logits_stride_j = logits.stride(2)  # should be 1

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _compute_logits_per_j_kernel[grid](
                q_slice, k_expanded, logits,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h,
                logits_stride_q, logits_stride_h, logits_stride_j,
                sm_scale,
            )

            # Compute lse per (i, h) using Triton reduction
            lse_slice = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)
            _lse_reduce_kernel[grid](
                logits, lse_slice,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                lse_slice.stride(0), lse_slice.stride(1),
            )

            # Accumulate output per (i, h) using Triton softmax*attention accumulation
            out_acc = torch.zeros((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            out_stride_b = out_acc.stride(0) * head_dim * out_acc.shape[1]
            out_stride_h = out_acc.stride(1) * head_dim
            out_stride_d = out_acc.stride(2)

            v_stride_b = v_expanded.stride(0) * head_dim * v_expanded.shape[1]  # stride for N
            v_stride_h = v_expanded.stride(1) * head_dim                       # stride for H
            v_stride_d = v_expanded.stride(2)                                  # stride for D (should be 1)

            _softmax_accum_output_kernel[grid](
                logits, lse_slice, v_expanded, out_acc,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                v_stride_b, v_stride_h, v_stride_d,
                out_stride_b, out_stride_h, out_stride_d,
            )

            # Store results into output for this b-slice (cast to bfloat16)
            output[q_start:q_end] = out_acc.to(torch.bfloat16)
            # Store lse for this b-slice (float32)
            lse[q_start:q_end] = lse_slice

        return output, lse


def run(*args):
    return ModelNew()(*args)
