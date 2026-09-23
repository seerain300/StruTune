import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    q_ptr,              # *float32, shape [num_q_tokens, 32, 128]
    k_ptr,              # *float32, shape [num_kv_tokens, 32, 128] (expanded from 8)
    logits_ptr,         # *float32, shape [num_q_tokens, num_kv_tokens] per (i, h)
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    head_dim,           # int
    sm_scale,           # float32 scalar
    q_stride_b, q_stride_h, q_stride_d,   # strides for q
    k_stride_b, k_stride_h, k_stride_d,   # strides for k
    log_stride_i, log_stride_j,           # strides for logits (we pass 2D strides; here 1D view per (i,h))
    pid: tl.constexpr,                    # program id in [0, num_q_tokens*32)
):
    # One program per (i, h)
    i = pid // 32
    h = pid % 32

    # Guard in case pid exceeds num_q_tokens*32 (in case grid is off)
    if i >= num_q_tokens:
        return

    # Pointers to q[i, h, :] and k[:, h, :]
    q_vec_ptr = q_ptr + i * q_stride_b + h * q_stride_h
    # We'll iterate j over k, load k[j, h, :]
    for j in range(0, num_kv_tokens):
        k_vec_ptr = k_ptr + j * k_stride_b + h * k_stride_h  # h is in [0,32]; k has 32 heads after expand

        # Load scalars
        q_val = tl.load(q_vec_ptr)  # q[i, h]
        k_val = tl.load(k_vec_ptr)  # k[j, h]

        score = q_val * k_val * sm_scale
        # Causal mask: i can attend j < (i + 1 + delta). Here delta = num_kv_tokens - num_q_tokens.
        delta = num_kv_tokens - num_q_tokens
        valid = j < (i + 1 + delta)
        score = tl.where(valid, score, -float('inf'))

        # Store to logits[i, j] for this (i, h)
        logits_off = i * log_stride_i + j * log_stride_j
        tl.store(logits_ptr + logits_off, score)


@triton.jit
def _lse_kernel(
    logits_ptr,         # *float32, shape [num_q_tokens, num_kv_tokens], viewed per (i,h)
    lse_ptr,            # *float32, shape [num_q_tokens, 32]
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    log_stride_i, log_stride_j,           # strides for logits
    lse_stride_i, lse_stride_h,           # strides for lse
    pid: tl.constexpr,                    # program id in [0, num_q_tokens*32)
):
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Compute max over j for numerical stability
    m = -float('inf')
    for j in range(0, num_kv_tokens):
        off = i * log_stride_i + j * log_stride_j
        val = tl.load(logits_ptr + off)
        if val > m:
            m = val

    # Compute sum exp(logits - m)
    s = 0.0
    for j in range(0, num_kv_tokens):
        off = i * log_stride_i + j * log_stride_j
        val = tl.load(logits_ptr + off)
        s += tl.exp(val - m)

    lse_val = tl.log(s) + m  # standard logsumexp; original code uses logsumexp, not divided by log(2)

    # Store lse[i, h]
    tl.store(lse_ptr + i * lse_stride_i + h * lse_stride_h, lse_val)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr,         # *float32, shape [num_q_tokens, num_kv_tokens] per (i,h)
    v_ptr,              # *float32, shape [num_kv_tokens, 32, 128] (expanded from 8)
    output_ptr,         # *float32, shape [num_q_tokens, 32, 128]
    lse_ptr,            # *float32, shape [num_q_tokens, 32]
    num_q_tokens: tl.constexpr,
    num_kv_tokens: tl.constexpr,
    head_dim: tl.constexpr,               # 128
    lse_stride_i, lse_stride_h,           # strides for lse
    out_stride_b, out_stride_h, out_stride_d,  # strides for output
    v_stride_b, v_stride_h, v_stride_d,       # strides for v (32 heads)
    pid: tl.constexpr,                    # program id in [0, num_q_tokens*32)
):
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_val = tl.load(lse_ptr + i * lse_stride_i + h * lse_stride_h)

    # Accumulate output[i, h, :] across j
    # We'll use a vector d across head_dim for output stores, but accumulate scalar per d.
    for d in range(0, head_dim):
        acc = 0.0
        for j in range(0, num_kv_tokens):
            off_logits = i * log_stride_i + j * log_stride_j
            logit = tl.load(logits_ptr + off_logits)  # scalar
            y = tl.exp(logit - lse_val)              # scalar
            # v[j, h, d] pointer: v_ptr + j*v_stride_b + h*v_stride_h + d*v_stride_d
            v_ptr_jhd = v_ptr + j * v_stride_b + h * v_stride_h + d * v_stride_d
            v_val = tl.load(v_ptr_jhd)              # scalar
            acc += y * v_val

        # Store acc into output[i, h, d]
        out_ptr_d = output_ptr + i * out_stride_b + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Nothing to initialize

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA for Triton kernels."
        device = q.device

        # Number of segments
        len_indptr = qo_indptr.shape[0]
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())

        # Output and LSE buffers
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)  # we'll convert to bfloat16 at end
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Process each segment b
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice q, k, v
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # Convert to float32 for computation
            q_f32 = q_batch.to(torch.float32).contiguous()
            k_f32 = k_batch.to(torch.float32).contiguous()
            v_f32 = v_batch.to(torch.float32).contiguous()

            # Expand k and v to 32 heads
            # Note: k_f32 has shape [num_kv_tokens, 8, 128]; we need 32 heads -> repeat_interleave along head dim
            k_expanded = k_f32.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_f32.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits buffer [num_q_tokens, num_kv_tokens] per (i, h)
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch kernel to compute logits[i, h, j] for all (i,h) and j
            # Grid: one program per (i, h) = num_q_tokens * 32
            grid = (num_q_tokens * 32,)
            # Strides for q
            q_stride_b = q_f32.stride(0) * 32 * 128  # not used directly; we use pointer arithmetic
            q_stride_h = q_f32.stride(1) * 128
            q_stride_d = q_f32.stride(2)
            # Strides for k (expanded)
            k_stride_b = k_expanded.stride(0) * 32 * 128
            k_stride_h = k_expanded.stride(1) * 128
            k_stride_d = k_expanded.stride(2)
            # Strides for logits viewed as [num_q_tokens, num_kv_tokens]
            log_stride_i = num_kv_tokens
            log_stride_j = 1
            _compute_logits_kernel[grid](
                q_f32, k_expanded, logits,
                num_q_tokens, num_kv_tokens, 128, sm_scale,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                log_stride_i, log_stride_j,
                0,  # pid is passed as constexpr via meta? Triton requires explicit meta; we pass as normal arg
            )
            # Note: Triton requires program_id to be explicit; we instead pass a simple grid and compute pid via tl.program_id(0).
            # Adjust kernel to read program_id from tl.program_id(0) and use it.

            # Fix: redefine compute_logits with program_id usage:
            # Triton requires passing grid as (grid,) and using tl.program_id(0) inside. We will redefine with that usage.

            # We need to re-run with proper kernel using tl.program_id(0). Let's define it correctly and relaunch.

            # Rerun with correct program_id usage in kernel
            # Define a corrected kernel signature:
            @triton.jit
            def _compute_logits_kernel(
                q_ptr, k_ptr, logits_ptr,
                num_q_tokens, num_kv_tokens, head_dim, sm_scale,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                log_stride_i, log_stride_j,
            ):
                pid = tl.program_id(axis=0)
                i = pid // 32
                h = pid % 32
                if i >= num_q_tokens:
                    return
                q_vec_ptr = q_ptr + i * q_stride_b + h * q_stride_h
                for j in range(0, num_kv_tokens):
                    k_vec_ptr = k_ptr + j * k_stride_b + h * k_stride_h
                    q_val = tl.load(q_vec_ptr)
                    k_val = tl.load(k_vec_ptr)
                    score = q_val * k_val * sm_scale
                    delta = num_kv_tokens - num_q_tokens
                    valid = j < (i + 1 + delta)
                    score = tl.where(valid, score, -float('inf'))
                    tl.store(logits_ptr + i * log_stride_i + j * log_stride_j, score)

            grid = (num_q_tokens * 32,)
            _compute_logits_kernel[grid](
                q_f32, k_expanded, logits,
                num_q_tokens, num_kv_tokens, 128, sm_scale,
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                num_kv_tokens, 1,
            )

            # Now compute LSE[i, h]
            @triton.jit
            def _lse_kernel(
                logits_ptr, lse_ptr,
                num_q_tokens, num_kv_tokens, log_stride_i, log_stride_j,
                lse_stride_i, lse_stride_h,
            ):
                pid = tl.program_id(axis=0)
                i = pid // 32
                h = pid % 32
                if i >= num_q_tokens:
                    return
                m = -float('inf')
                for j in range(0, num_kv_tokens):
                    val = tl.load(logits_ptr + i * log_stride_i + j * log_stride_j)
                    if val > m:
                        m = val
                s = 0.0
                for j in range(0, num_kv_tokens):
                    val = tl.load(logits_ptr + i * log_stride_i + j * log_stride_j)
                    s += tl.exp(val - m)
                lse_val = tl.log(s) + m
                tl.store(lse_ptr + i * lse_stride_i + h * lse_stride_h, lse_val)

            grid = (num_q_tokens * 32,)
            lse_stride_i = total_q  # not used; we can set lse.stride(0) but Triton uses strides from tensor, we pass element-wise strides
            lse_stride_h = 32       # similarly, set logical strides
            # lse tensor strides:
            lse_stride_i = lse.stride(0)
            lse_stride_h = lse.stride(1)
            _lse_kernel[grid](
                logits, lse,
                num_q_tokens, num_kv_tokens, num_kv_tokens, 1,
                lse_stride_i, lse_stride_h,
            )

            # Finally, compute output[i, h, :] = sum_j exp(logits[i, h, j] - lse[i, h]) * v_expanded[j, h, :]
            @triton.jit
            def _softmax_accum_output_kernel(
                logits_ptr, v_ptr, output_ptr, lse_ptr,
                num_q_tokens, num_kv_tokens, head_dim,
                lse_stride_i, lse_stride_h,
                out_stride_b, out_stride_h, out_stride_d,
                v_stride_b, v_stride_h, v_stride_d,
            ):
                pid = tl.program_id(axis=0)
                i = pid // 32
                h = pid % 32
                if i >= num_q_tokens:
                    return
                lse_val = tl.load(lse_ptr + i * lse_stride_i + h * lse_stride_h)
                for d in range(0, head_dim):
                    acc = 0.0
                    for j in range(0, num_kv_tokens):
                        logit = tl.load(logits_ptr + i * log_stride_i + j * log_stride_j)
                        y = tl.exp(logit - lse_val)
                        v_ptr_jhd = v_ptr + j * v_stride_b + h * v_stride_h + d * v_stride_d
                        v_val = tl.load(v_ptr_jhd)
                        acc += y * v_val
                    out_ptr_d = output_ptr + i * out_stride_b + h * out_stride_h + d * out_stride_d
                    tl.store(out_ptr_d, acc)

            grid = (num_q_tokens * 32,)
            out_stride_b = output.stride(0) * 32 * 128
            out_stride_h = output.stride(1) * 128
            out_stride_d = output.stride(2)
            v_stride_b = v_expanded.stride(0) * 32 * 128
            v_stride_h = v_expanded.stride(1) * 128
            v_stride_d = v_expanded.stride(2)
            _softmax_accum_output_kernel[grid](
                logits, v_expanded, output, lse,
                num_q_tokens, num_kv_tokens, 128,
                lse_stride_i, lse_stride_h,
                out_stride_b, out_stride_h, out_stride_d,
                v_stride_b, v_stride_h, v_stride_d,
            )

        # Convert output to bfloat16 to match original return type and return (output, lse)
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
