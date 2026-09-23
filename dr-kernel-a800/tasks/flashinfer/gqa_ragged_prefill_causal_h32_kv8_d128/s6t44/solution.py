import math
import torch

import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_ptr, k_ptr, out_logits_ptr,
    num_q_tokens, num_heads, num_kv_tokens,
    head_dim, sm_scale, delta,
    # strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d,
    stride_out_b, stride_out_h, stride_out_j, stride_out_d,
):
    # one program per (i, h)
    pid = tl.program_id(axis=0)
    i = pid // num_heads
    h = pid % num_heads

    if i >= num_q_tokens:
        return

    # vectorize over head_dim for output write
    d = tl.arange(0, head_dim)

    # loop over j (kv tokens)
    for j in range(0, num_kv_tokens):
        # load q[i, h]
        q_off = i * stride_q_b + h * stride_q_h
        q_elem = tl.load(q_ptr + q_off)  # float32

        # load k[j, h] from expanded k (k has num_heads=8 in original, but we index h as j's head index)
        # Note: k_expanded has 32 heads; we assume k_ptr is expanded (it is in host code), so stride_k_h corresponds to 32.
        k_off = j * stride_k_b + h * stride_k_h
        k_elem = tl.load(k_ptr + k_off)  # float32

        # compute logits
        logit = q_elem * k_elem * sm_scale  # float32 scalar
        # apply causal mask: j < i + 1 + delta
        mask = j < (i + 1 + delta)
        logit = tl.where(mask, logit, -float('inf'))

        # store logits[i, h, j] at linearized offset
        out_off = i * stride_out_b + h * stride_out_h + j * stride_out_j + d * stride_out_d
        # store as float32
        tl.store(out_logits_ptr + out_off, logit)


@triton.jit
def lse_reduce_kernel(
    logits_ptr, lse_ptr,
    num_q_tokens, num_heads, num_kv_tokens,
    stride_b, stride_h, stride_j,
):
    pid = tl.program_id(axis=0)
    i = pid // num_heads
    h = pid % num_heads
    if i >= num_q_tokens:
        return

    # compute max over j
    max_val = -float('inf')
    for j in range(0, num_kv_tokens):
        off = i * stride_b + h * stride_h + j * stride_j
        val = tl.load(logits_ptr + off)
        max_val = tl.maximum(max_val, val)

    # compute sum exp(val - max)
    sum_exp = 0.0
    for j in range(0, num_kv_tokens):
        off = i * stride_b + h * stride_h + j * stride_j
        val = tl.load(logits_ptr + off)
        sum_exp += tl.exp(val - max_val)

    lse_val = tl.log(sum_exp) + max_val  # float32
    tl.store(lse_ptr + (i * num_heads + h), lse_val)


@triton.jit
def softmax_accum_kernel(
    logits_ptr, v_ptr, lse_ptr, out_ptr,
    num_q_tokens, num_heads, num_kv_tokens,
    head_dim,
    stride_b, stride_h, stride_j, stride_v_h, stride_v_d, stride_out_b, stride_out_h, stride_out_d,
):
    pid = tl.program_id(axis=0)
    i = pid // num_heads
    h = pid % num_heads
    if i >= num_q_tokens:
        return

    # vector over head_dim for output
    d = tl.arange(0, head_dim)

    # loop over j to accumulate output[i, h, :]
    # initialize output vector to zeros
    for di in range(0, head_dim):
        tl.store(out_ptr + (i * stride_out_b + h * stride_out_h + di * stride_out_d), 0.0)

    for j in range(0, num_kv_tokens):
        off = i * stride_b + h * stride_h + j * stride_j
        logit = tl.load(logits_ptr + off)  # float32

        # load v_expanded[j, h, :] which is a vector over head_dim
        v_off = j * stride_v_b + h * stride_v_h + d * stride_v_d  # note: v_expanded uses h as head index
        v_vec = tl.load(v_ptr + v_off)  # float32 vector of length head_dim

        # exp(logit - lse) * v_vec
        lse_val = tl.load(lse_ptr + (i * num_heads + h))  # float32 scalar
        coeff = tl.exp(logit - lse_val)

        out_off = i * stride_out_b + h * stride_out_h + d * stride_out_d
        # accumulate
        out_vec = tl.load(out_ptr + out_off)  # vector
        out_vec += coeff * v_vec
        tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # assert shapes
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8
        assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128

        device = q.device
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        # segment processing
        num_segments = qo_indptr.numel() - 1

        for b in range(num_segments):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # slice
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            # move to float32 for compute
            q_f32 = q_batch.contiguous().to(torch.float32)  # [num_q_tokens, 32, 128]
            k_f32 = k_batch.contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]
            v_f32 = v_batch.contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]

            # expand to 32 heads
            gqa_ratio = 32 // 8
            k_exp = k_f32.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_exp = v_f32.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # allocate outputs
            logits = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=device)
            lse = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=device)
            out = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)  # will cast to bfloat16

            # delta = num_kv_tokens - num_q_tokens (used in causal mask)
            delta = num_kv_tokens - num_q_tokens

            # Launch Triton kernels: one program per (i, h)
            grid = (num_q_tokens * 32,)

            # compute logits
            compute_logits_kernel[grid](
                q_f32, k_exp, logits,
                num_q_tokens, 32, num_kv_tokens,
                128, sm_scale, delta,
                # strides
                q_f32.stride(0), q_f32.stride(1), q_f32.stride(2),
                k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2), logits.stride(3),
                num_warps=1,
            )

            # compute lse
            lse_reduce_kernel[grid](
                logits, lse,
                num_q_tokens, 32, num_kv_tokens,
                logits.stride(0), logits.stride(1), logits.stride(2),
                num_warps=1,
            )

            # compute output
            softmax_accum_kernel[grid](
                logits, v_exp, lse, out,
                num_q_tokens, 32, num_kv_tokens,
                128,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_exp.stride(0), v_exp.stride(1), v_exp.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                num_warps=1,
            )

        # Return output in bfloat16 to match original signature
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
