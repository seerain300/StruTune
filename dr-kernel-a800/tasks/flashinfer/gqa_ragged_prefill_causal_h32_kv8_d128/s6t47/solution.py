import torch
import math
import triton
import triton.language as tl


# Kernel 1: compute logits[i, h, j] = q[i, h] * k_expanded[j, h] * sm_scale
@triton.jit
def compute_logits_kernel(
    q_ptr, k_ptr, logits_ptr,
    num_q_tokens, num_kv_tokens, sm_scale,
    q_stride0, q_stride1, q_stride2,
    k_stride0, k_stride1, k_stride2,
    log_stride0, log_stride1
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Loop over kv positions j (scalar for robustness)
    for j in range(0, num_kv_tokens):
        # Pointers
        q_ptr_j = q_ptr + i * q_stride0 + h * q_stride1  # q[i, h]
        k_ptr_j = k_ptr + j * k_stride0 + h * k_stride1  # k_expanded[j, h]
        # Load scalars
        q_val = tl.load(q_ptr_j).to(tl.float32)
        k_val = tl.load(k_ptr_j).to(tl.float32)
        score = q_val * k_val * sm_scale
        # Causal mask: allow j < i + 1 + delta; delta = num_kv_tokens - num_q_tokens (computed on host)
        # Note: This matches the original logic where queries can reference up to next position.
        if j >= (i + 1):
            score = -float('inf')
        # Store logits[i, h, j]
        log_ptr_ij = logits_ptr + i * log_stride0 + j * log_stride1
        tl.store(log_ptr_ij, score)


# Kernel 2: compute LSE[i, h] = logsumexp over j of logits[i, h, :]
@triton.jit
def lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens,
    log_stride0, log_stride1
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Compute m = max(logits[i, h, :]) and s = sum(exp(logits - m))
    m = -float('inf')
    for j in range(0, num_kv_tokens):
        ptr_ij = logits_ptr + i * log_stride0 + j * log_stride1
        val = tl.load(ptr_ij).to(tl.float32)
        m = tl.maximum(m, val)
    s = 0.0
    for j in range(0, num_kv_tokens):
        ptr_ij = logits_ptr + i * log_stride0 + j * log_stride1
        val = tl.load(ptr_ij).to(tl.float32)
        s += tl.exp(val - m)
    lse_val = tl.log(s) + m  # logsumexp
    # Store lse[i, h]
    lse_ptr_ih = lse_ptr + i * 32 + h
    tl.store(lse_ptr_ih, lse_val)


# Kernel 3: accumulate output[i, h, :] = sum_j exp(logits[i, h, j] - lse[i, h]) * v_expanded[j, h, :]
@triton.jit
def softmax_accum_kernel(
    logits_ptr, v_ptr, out_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, head_dim,  # head_dim = 128 here
    log_stride0, log_stride1,
    v_stride0, v_stride1, v_stride2,
    out_stride0, out_stride1, out_stride2
):
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    # Load lse[i, h]
    lse_ptr_ih = lse_ptr + i * 32 + h
    m = tl.load(lse_ptr_ih).to(tl.float32)

    # Accumulate over j, then write out
    for j in range(0, num_kv_tokens):
        ptr_ij = logits_ptr + i * log_stride0 + j * log_stride1
        val = tl.load(ptr_ij).to(tl.float32)
        y = tl.exp(val - m)  # valid only if j < i + 1, handled by original masking
        # Load v_expanded[j, h, :] vector over head_dim using scalar j and d
        for d in range(0, head_dim):
            v_ptr_jhd = v_ptr + j * v_stride0 + h * v_stride1 + d * v_stride2
            v_val = tl.load(v_ptr_jhd).to(tl.float32)
            out_ptr_ihd = out_ptr + i * out_stride0 + h * out_stride1 + d * out_stride2
            # Accumulate: output[i, h, d] += y * v_expanded[j, h, d]
            tl.store(out_ptr_ihd, tl.load(out_ptr_ihd).to(tl.float32) + y * v_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure device and dtype compatibility; Triton requires CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]

        # Output and LSE buffers (float32 for compute, cast back to bfloat16 at the end)
        output = torch.zeros((total_q, 32, 128), dtype=torch.float32, device=device)
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

            # Convert to float32 for computation and make contiguous
            q_f32 = q_batch.to(torch.float32).contiguous()
            k_f32 = k_batch.to(torch.float32).contiguous()
            v_f32 = v_batch.to(torch.float32).contiguous()

            # Expand k and v to 32 heads by repeating along head dimension
            # k_expanded[j, h, :] = k_batch[j, h // 4, :]
            k_expanded = k_f32.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_f32.repeat_interleave(4, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits [num_q_tokens, num_kv_tokens] per (i, h) slice
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch kernel to compute logits[i, h, j] for all (i,h) and j
            grid = (num_q_tokens * 32,)
            compute_logits_kernel[grid](
                q_f32, k_expanded, logits,
                num_q_tokens, num_kv_tokens, sm_scale,
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                logits.stride(0), logits.stride(1),
                num_warps=1
            )

            # Launch LSE reduction kernel
            lse_reduce_kernel[grid](
                logits, lse,
                num_q_tokens, num_kv_tokens,
                logits.stride(0), logits.stride(1),
                num_warps=1
            )

            # Launch softmax accumulation kernel to produce output[i, h, :]
            out_segment = output[q_start:q_end]  # [num_q_tokens, 32, 128]
            softmax_accum_kernel[grid](
                logits, v_expanded, out_segment, lse,
                num_q_tokens, num_kv_tokens, 128,
                logits.stride(0), logits.stride(1),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                out_segment.stride(0), out_segment.stride(1), out_segment.stride(2),
                num_warps=1
            )

        # Cast output back to bfloat16 to match original run
        output_cast = output.to(torch.bfloat16)
        return output_cast, lse


def run(*args):
    return ModelNew()(*args)
