import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_per_j_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    scale: tl.float32,
):
    # One program per (query, head)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Loop over kv positions and compute q[i, h] * k[j, h] * scale, store in logits[i, h, j]
    for j in range(0, num_kv_tokens):
        # q has shape [num_q_tokens, num_qo_heads, head_dim], full and contiguous.
        q_offset = i * (num_qo_heads * head_dim) + h * head_dim
        q_val = tl.load(q_ptr + q_offset)

        # k has shape [num_kv_tokens, num_qo_heads, head_dim], full and contiguous (we'll ensure .contiguous()).
        k_offset = j * (num_qo_heads * head_dim) + h * head_dim
        k_val = tl.load(k_ptr + k_offset)

        # Compute score
        score = q_val * k_val * scale

        # Causal mask: i can only attend to j < i + 1 + delta
        delta = num_kv_tokens - num_q_tokens  # runtime scalar
        causal = j < (i + 1 + delta)
        score = tl.where(causal, score, -float("inf"))

        # Store logits[i, h, j] in float32 buffer of shape [num_q_tokens, num_qo_heads, num_kv_tokens]
        # logits layout: contiguous [N, H, T], so element offset = i*(H*T) + h*T + j
        logits_offset = i * (num_qo_heads * num_kv_tokens) + h * num_kv_tokens + j
        tl.store(logits_ptr + logits_offset, score)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    lse_stride_b, lse_stride_h,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute logsumexp along j for (i, h): lse[i, h] = log(sum(exp(score - m))) + m
    # Pass 1: m = max(score)
    m = -float("inf")
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        m = tl.maximum(m, score)

    # Pass 2: s = sum(exp(score - m))
    s = 0.0
    for j in range(0, num_kv_tokens):
        offset = i * lse_stride_b + h * lse_stride_h + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)
        s += y

    # Store lse[i, h] = log(s) + m
    lse_offset = i * lse_stride_b + h * lse_stride_h
    tl.store(lse_ptr + lse_offset, tl.log(s) + m)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr, lse_ptr, v_expanded_ptr, out_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads,
    out_stride_b, out_stride_h, out_stride_d,
    lse_stride_b, lse_stride_h,
):
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_offset = i * lse_stride_b + h * lse_stride_h
    m = tl.load(lse_ptr + lse_offset)

    # Accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - m) * v_expanded[j, h, :]
    out_base = i * out_stride_b + h * out_stride_h
    d = tl.arange(0, head_dim)
    out_ptr_vec = out_ptr + out_base + d

    for j in range(0, num_kv_tokens):
        # score = logits[i, h, j]
        offset = i * (num_qo_heads * num_kv_tokens) + h * num_kv_tokens + j
        score = tl.load(logits_ptr + offset)
        y = tl.exp(score - m)

        # v_expanded[j, h, :] is [head_dim], contiguous along d; v_expanded has shape [num_kv_tokens, num_qo_heads, head_dim]
        v_base = j * (num_qo_heads * head_dim) + h * head_dim
        v_ptr_vec = v_expanded_ptr + v_base + d
        val = y * tl.load(v_ptr_vec)

        # out[i, h, :] += val
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

        # Total counts (as in original checks)
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

            # Use full, contiguous q/k/v to avoid strided pointer issues in kernels
            q_full = q.contiguous()
            k_full = k.contiguous()
            v_full = v.contiguous()

            # Slice pointers (not tensors) for this segment
            q_slice_ptr = q_full[q_start:q_end]            # view; we will index via pointer arithmetic
            k_slice_ptr = k_full[kv_start:kv_end]          # same
            v_slice_ptr = v_full[kv_start:kv_end]          # same

            # Expand K/V to 32 heads (GQA)
            k_expanded = k_slice_ptr.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
            v_expanded = v_slice_ptr.repeat_interleave(gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_slice_ptr.shape[0]
            num_kv_tokens = k_expanded.shape[0]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens], float32, contiguous
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per (i, h)
            grid = (num_q_tokens * num_qo_heads,)
            _compute_logits_per_j_kernel[grid](
                q_full, k_expanded, logits,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
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
            # Correct stride computation for 3D tensor
            out_stride_b = out_acc.stride(0)   # element stride for batch
            out_stride_h = out_acc.stride(1)   # element stride for heads
            out_stride_d = out_acc.stride(2)   # element stride for features (should be 1)

            _softmax_accum_output_kernel[grid](
                logits, lse_slice, v_expanded, out_acc,
                num_q_tokens, num_kv_tokens, num_qo_heads,
                out_stride_b, out_stride_h, out_stride_d,
                lse_slice.stride(0), lse_slice.stride(1),
            )

            # Store results into output for this b-slice (cast to bfloat16)
            output[q_start:q_end] = out_acc.to(torch.bfloat16)
            # Store lse for this b-slice (float32)
            lse[q_start:q_end] = lse_slice

        return output, lse


def run(*args):
    return ModelNew()(*args)
