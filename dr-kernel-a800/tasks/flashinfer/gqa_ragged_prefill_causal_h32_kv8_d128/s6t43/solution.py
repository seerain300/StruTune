import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_qo_heads, num_kv_tokens, sm_scale,
    stride_q_b, stride_q_h, stride_k_h, stride_log_b, stride_log_h, stride_log_j,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load q[i, h] as scalar float32
    q_base = q_ptr + i * stride_q_b + h * stride_q_h
    q_val = tl.load(q_base).to(tl.float32)

    # Loop over j and compute logits[i, h, j] = q_val * k_expanded[j, h] * sm_scale
    for j in range(0, num_kv_tokens):
        k_base = k_ptr + j * stride_k_h + h * stride_k_h
        k_val = tl.load(k_base).to(tl.float32)
        log_val = q_val * k_val * sm_scale  # scalar
        # Causal mask: j < i + 1
        if j >= (i + 1):
            log_val = -float("inf")
        # Store to logits[i, h, j]
        log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
        tl.store(log_base, log_val)


@triton.jit
def lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_qo_heads, num_kv_tokens,
    stride_log_b, stride_log_h, stride_log_j,
    stride_lse_b, stride_lse_h,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Compute logsumexp over j for logits[i, h, :]
    max_val = -float("inf")
    sum_val = 0.0
    for j in range(0, num_kv_tokens):
        log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
        val = tl.load(log_base)  # float32 scalar
        max_val = tl.maximum(max_val, val)
    # Now compute sum(exp(val - max_val))
    for j in range(0, num_kv_tokens):
        log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
        val = tl.load(log_base)
        sum_val += tl.exp(val - max_val)
    lse = tl.log(sum_val) + max_val  # standard logsumexp
    lse_base = lse_ptr + i * stride_lse_b + h * stride_lse_h
    tl.store(lse_base, lse)


@triton.jit
def softmax_accum_kernel(
    logits_ptr, v_ptr, lse_ptr, output_ptr,
    num_q_tokens, num_qo_heads, num_kv_tokens,
    stride_log_b, stride_log_h, stride_log_j,
    stride_v_h, stride_v_d, stride_out_b, stride_out_h, stride_out_d,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // num_qo_heads
    h = pid % num_qo_heads
    if i >= num_q_tokens:
        return

    # Load LSE[i, h]
    lse_base = lse_ptr + i * 0 + h * 0  # lse_ptr is [num_q_tokens, num_qo_heads], strides handled below
    # Adjust: lse_ptr is 2D, use strides properly
    lse_base = lse_ptr + i * 0 + h * 0  # incorrect, fix below

    # Fix: lse_ptr is [num_q_tokens, num_qo_heads], so:
    lse_base = lse_ptr + i * 0 + h * 0  # still incorrect; we need to pass strides for lse_ptr too.

    # To avoid confusion, re-express: lse_ptr has shape (num_q_tokens, num_qo_heads), we need to compute i and h.
    # We need to know strides for lse_ptr. Since we don't have them here, we pass them via lse_ptr's allocation:
    # lse_ptr is allocated as (num_q_tokens, num_qo_heads) with torch.empty(...), so we can use .stride().
    # Triton kernel requires stride arguments; we will pass stride_lse_b = lse_ptr.stride(0), stride_lse_h = lse_ptr.stride(1).
    # However, Triton does not accept runtime attributes here. So we need to pass them as constexpr meta parameters.
    # To keep code simple, we will pass strides for lse_ptr as kwargs in the call site.

    # Therefore, in order to correctly address lse_ptr, we must pass stride_lse_b and stride_lse_h to the kernel.
    # Since Triton doesn't allow arbitrary kwargs, we will pass them as constexpr meta arguments when launching.
    # For this, we redefine the kernel signature to include them. We'll adjust the call site accordingly.

    # Since we can't change signature after, we will instead compute lse directly on host after kernel 2. But we need to
    # keep everything in Triton. So we will store LSE in a separate kernel. To access it in this kernel, we will
    # recompute LSE here using tl.load with correct strides.

    # Compute LSE[i, h] by loading it from lse_ptr with strides (num_q_tokens, num_qo_heads). Triton requires stride
    # arguments; we will pass them as constexpr via the launch. For clarity, we re-launch a tiny helper to load LSE.
    # Triton does not support dynamic loading of LSE here; thus we will recompute LSE in this kernel via two passes.

    # First, recompute max and sum for logsumexp over j for logits[i, h, :].
    max_val = -float("inf")
    sum_val = 0.0
    for j in range(0, num_kv_tokens):
        log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
        val = tl.load(log_base)
        max_val = tl.maximum(max_val, val)
    for j in range(0, num_kv_tokens):
        log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
        val = tl.load(log_base)
        sum_val += tl.exp(val - max_val)
    lse_val = tl.log(sum_val) + max_val

    # Now accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - lse_val) * v_expanded[j, h, :]
    # Causal mask: j < i + 1
    for d in range(0, 128):  # head_dim is 128
        # We need v[j, h, d] for all j contributing (j < i + 1)
        contrib = 0.0
        for j in range(0, num_kv_tokens):
            if j < (i + 1):
                log_base = logits_ptr + i * stride_log_b + h * stride_log_h + j * stride_log_j
                val = tl.load(log_base)
                # exp(val - lse_val) contribution
                exp_val = tl.exp(val - lse_val)
                v_base = v_ptr + j * stride_v_h + h * stride_v_h + d * stride_v_d
                v_val = tl.load(v_base)
                contrib += exp_val * v_val
        out_base = output_ptr + i * stride_out_b + h * stride_out_h + d * stride_out_d
        tl.store(out_base, contrib)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Extract shapes
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert qo_indptr.dtype == torch.int32 and kv_indptr.dtype == torch.int32
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Iterate over segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slicing and expansion (contiguous)
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)       # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)     # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)     # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]

            # Expand to 32 heads
            k_expanded = k_batch.repeat_interleave(4, dim=1)               # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)               # [num_kv_tokens, 32, 128]

            # Allocate logits
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Launch kernel to compute logits[i, h, j] = q[i, h] * k_expanded[j, h] * sm_scale
            grid = (num_q_tokens * num_qo_heads,)
            # Strides:
            stride_q_b, stride_q_h, stride_q_d = q_batch.stride()  # (N, H, D) -> (H*128, 128, 1)
            stride_k_h, stride_k_d = k_expanded.stride(1), k_expanded.stride(2)  # (H, D)
            stride_log_b, stride_log_h, stride_log_j = logits.stride()  # (N, H, J)

            compute_logits_kernel[grid](
                q_batch, k_expanded, logits,
                num_q_tokens, num_qo_heads, num_kv_tokens, sm_scale,
                stride_q_b, stride_q_h, stride_k_h, stride_log_b, stride_log_h, stride_log_j,
                num_warps=1, num_stages=1
            )

            # Launch kernel to compute LSE[i, h] = logsumexp(logits[i, h, :])
            lse_ptr = lse[q_start:q_end]  # already allocated
            stride_lse_b, stride_lse_h = lse_ptr.stride()  # (num_q_tokens, num_qo_heads) strides
            lse_reduce_kernel[grid](
                logits, lse_ptr,
                num_q_tokens, num_qo_heads, num_kv_tokens,
                stride_log_b, stride_log_h, stride_log_j,
                stride_lse_b, stride_lse_h,
                num_warps=1, num_stages=1
            )

            # Launch kernel to compute output[i, h, :] = sum_j exp(logits[i, h, j] - LSE[i, h]) * v_expanded[j, h, :]
            output_slice = output[q_start:q_end]  # [num_q_tokens, 32, 128], bfloat16
            stride_out_b, stride_out_h, stride_out_d = output_slice.stride()
            stride_v_h, stride_v_d = v_expanded.stride(1), v_expanded.stride(2)
            softmax_accum_kernel[grid](
                logits, v_expanded, lse_ptr, output_slice,
                num_q_tokens, num_qo_heads, num_kv_tokens,
                stride_log_b, stride_log_h, stride_log_j,
                stride_v_h, stride_v_d, stride_out_b, stride_out_h, stride_out_d,
                num_warps=1, num_stages=1
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
