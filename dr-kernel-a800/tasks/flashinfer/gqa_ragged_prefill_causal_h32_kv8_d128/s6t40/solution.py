import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    q_ptr,  # *f32 [num_q_tokens, 32, 128]
    k_ptr,  # *f32 [num_kv_tokens, 32, 128] (k_expanded)
    logits_ptr,  # *f32 [num_q_tokens, 32, num_kv_tokens]
    sm_scale: tl.float32,
    num_q_tokens: tl.int32,
    num_kv_tokens: tl.int32,
    q_stride_b: tl.int32, q_stride_h: tl.int32, q_stride_d: tl.int32,
    k_stride_b: tl.int32, k_stride_h: tl.int32, k_stride_d: tl.int32,
    logits_stride_b: tl.int32, logits_stride_h: tl.int32, logits_stride_j: tl.int32,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    q_base = i * q_stride_b + h * q_stride_h
    # Iterate over kv positions j
    for j in range(0, num_kv_tokens):
        k_base = j * k_stride_b + h * k_stride_h
        q_val = tl.load(q_ptr + q_base)  # scalar load
        k_val = tl.load(k_ptr + k_base)  # scalar load
        score = q_val * k_val * sm_scale
        # Simple causal mask: allow j < i + 1
        allowed = j < (i + 1)
        score = tl.where(allowed, score, -float('inf'))
        logits_off = i * logits_stride_b + h * logits_stride_h + j * logits_stride_j
        tl.store(logits_ptr + logits_off, score)


@triton.jit
def _lse_reduce_kernel(
    logits_ptr,  # *f32 [num_q_tokens, 32, num_kv_tokens]
    lse_ptr,     # *f32 [num_q_tokens, 32]
    num_q_tokens: tl.int32,
    num_kv_tokens: tl.int32,
    logits_stride_b: tl.int32, logits_stride_h: tl.int32, logits_stride_j: tl.int32,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    m = -float('inf')
    for j in range(0, num_kv_tokens):
        logits_off = i * logits_stride_b + h * logits_stride_h + j * logits_stride_j
        val = tl.load(logits_ptr + logits_off)
        m = tl.maximum(m, val)

    s = 0.0
    for j in range(0, num_kv_tokens):
        logits_off = i * logits_stride_b + h * logits_stride_h + j * logits_stride_j
        val = tl.load(logits_ptr + logits_off)
        s += tl.exp(val - m)

    lse = tl.log(s) + m
    lse_off = i * 32 + h  # lse is [num_q_tokens, 32] contiguous
    tl.store(lse_ptr + lse_off, lse)


@triton.jit
def _softmax_accum_output_kernel(
    logits_ptr,          # *f32 [num_q_tokens, 32, num_kv_tokens]
    v_ptr,               # *f32 [num_kv_tokens, 32, 128] (v_expanded)
    output_ptr,          # *f32 [num_q_tokens, 32, 128]
    lse_ptr,             # *f32 [num_q_tokens, 32]
    sm_scale: tl.float32,  # not used here
    num_q_tokens: tl.int32,
    num_kv_tokens: tl.int32,
    head_dim: tl.int32,    # 128
    logits_stride_b: tl.int32, logits_stride_h: tl.int32, logits_stride_j: tl.int32,
    v_stride_b: tl.int32, v_stride_h: tl.int32, v_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_h: tl.int32, out_stride_d: tl.int32,
):
    # One program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // 32
    h = pid % 32
    if i >= num_q_tokens:
        return

    lse_off = i * 32 + h
    m = tl.load(lse_ptr + lse_off)

    out_base = i * out_stride_b + h * out_stride_h
    # Initialize output vector to zero
    for d in range(0, head_dim):
        tl.store(output_ptr + out_base + d * out_stride_d, 0.0)

    # Accumulate over j with causal mask j < i + 1
    for j in range(0, num_kv_tokens):
        logits_off = i * logits_stride_b + h * logits_stride_h + j * logits_stride_j
        score = tl.load(logits_ptr + logits_off)
        if j < (i + 1):
            y = tl.exp(score - m)
            v_base = j * v_stride_b + h * v_stride_h
            for d in range(0, head_dim):
                v_val = tl.load(v_ptr + v_base + d * v_stride_d)
                out_off = out_base + d * out_stride_d
                curr = tl.load(output_ptr + out_off)
                curr += y * v_val
                tl.store(output_ptr + out_off, curr)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA device"
        # The original code uses bfloat16 inputs; we will compute in float32 for stability
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32
        assert num_kv_heads == 8

        output = torch.empty((total_q, 32, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice and convert to float32 for compute
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)          # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)        # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)        # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]

            # Expand K and V to 32 heads
            k_expanded = k_batch.repeat_interleave(4, dim=1)   # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(4, dim=1)   # [num_kv_tokens, 32, 128]

            # Allocate logits
            logits = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)

            # Strides
            q_stride_b, q_stride_h, q_stride_d = q_batch.stride()
            k_stride_b, k_stride_h, k_stride_d = k_expanded.stride()
            logits_stride_b, logits_stride_h, logits_stride_j = logits.stride()

            # Compute logits: one program per (i, h)
            grid = (num_q_tokens * 32,)
            _compute_logits_kernel[grid](
                q_batch, k_expanded, logits,
                sm_scale,
                num_q_tokens, num_kv_tokens,
                q_stride_b, q_stride_h, q_stride_d,
                k_stride_b, k_stride_h, k_stride_d,
                logits_stride_b, logits_stride_h, logits_stride_j,
                num_warps=1, num_stages=1,
            )

            # LSE reduction: one program per (i, h)
            l = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=q.device)  # dummy, will be replaced
            lse_sub = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=q.device)
            _lse_reduce_kernel[grid](
                logits, lse_sub,
                num_q_tokens, num_kv_tokens,
                logits_stride_b, logits_stride_h, logits_stride_j,
                num_warps=1, num_stages=1,
            )
            lse[:num_q_tokens, :32] = lse_sub  # assign back

            # Output accumulation: one program per (i, h)
            out_stride_b, out_stride_h, out_stride_d = output.stride()
            v_stride_b, v_stride_h, v_stride_d = v_expanded.stride()

            _softmax_accum_output_kernel[grid](
                logits, v_expanded, output, lse,
                sm_scale,
                num_q_tokens, num_kv_tokens, 128,
                logits_stride_b, logits_stride_h, logits_stride_j,
                v_stride_b, v_stride_h, v_stride_d,
                out_stride_b, out_stride_h, out_stride_d,
                num_warps=1, num_stages=1,
            )

        # Cast output back to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
