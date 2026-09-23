import torch
import math

import triton
import triton.language as tl


@triton.jit
def _gqa_segment_forward(
    q_ptr,  # *float32, shape [Nq, 32, 128]
    k_ptr,  # *float32, shape [Nk, 8, 128]
    v_ptr,  # *float32, shape [Nk, 8, 128]
    out_ptr,  # *float32, shape [Nq, 32, 128]
    lse_ptr,  # *float32, shape [Nq, 32]
    q_start,  # int32
    kv_start,  # int32
    q_end,     # int32
    kv_end,    # int32
    sm_scale,  # float32
):
    # Constants
    D = 128
    Q_HEADS = 32
    K_HEADS = 8
    LN2 = 0.6931471805599453  # ln(2)

    Nq = q_end - q_start
    Nk = kv_end - kv_start
    delta = Nk - Nq

    # Iterate over each query token
    for i in range(0, Nq):
        # Prepare output row and lse vector for this query token
        out_row = tl.zeros((Q_HEADS, D), dtype=tl.float32)
        lse_vec = tl.zeros((Q_HEADS,), dtype=tl.float32)

        # Compute logits across 32 heads by looping over original kv heads j in 0..7
        for j in range(0, K_HEADS):
            # Compute dot products for all query heads h
            for h in range(0, Q_HEADS):
                orig_h = h % K_HEADS

                # Load Q[i, h, :]
                q_off = (i * (Q_HEADS * D)) + h * D
                q_vec = tl.load(q_ptr + q_off + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], float32

                # Load K[j, orig_h, :]
                k_off = (j * (K_HEADS * D)) + orig_h * D
                k_vec = tl.load(k_ptr + kv_start * (K_HEADS * D) + k_off + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], float32

                dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
                logits_h = dot * sm_scale

                # Apply causal mask: allow only if j < (i + 1 + delta)
                if j >= (i + 1 + delta):
                    logits_h = -float('inf')

                lse_vec[h] = logits_h

        # Compute base-2 logsumexp over 32 logits
        max_logit = tl.max(lse_vec, axis=0)
        sum_exp = tl.sum(tl.exp(lse_vec - max_logit), axis=0)
        lse_base2 = (max_logit + tl.log(sum_exp)) / LN2

        # Softmax over 32 positions
        exp_vals = tl.exp(lse_vec - max_logit)
        sum_exp_all = tl.sum(exp_vals, axis=0)
        softmax_vals = exp_vals / sum_exp_all

        # Accumulate output: out_row[h, :] += softmax_vals[h] * v[j, orig_h, :]
        for j in range(0, K_HEADS):
            for h in range(0, Q_HEADS):
                orig_h = h % K_HEADS
                v_off = (j * (K_HEADS * D)) + orig_h * D
                v_vec = tl.load(v_ptr + kv_start * (K_HEADS * D) + v_off + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D], float32

                out_row[h, :] = out_row[h, :] + softmax_vals[h] * v_vec

        # Store output row
        base_out = (q_start + i) * (Q_HEADS * D)
        for h in range(0, Q_HEADS):
            tl.store(out_ptr + base_out + h * D + tl.arange(0, D), out_row[h, :], mask=tl.arange(0, D) < D)

        # Store lse vector for this query token
        lse_base = q_start + i
        for h in range(0, Q_HEADS):
            tl.store(lse_ptr + lse_base * Q_HEADS + h, lse_base2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        device = q.device

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Output and lse buffers
        output_f32 = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (total_q, num_qo_heads), dtype=torch.float32, device=device
        )

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slice and cast to float32 for computation
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)

            # Launch Triton kernel for this segment
            _gqa_segment_forward[(1,)](
                q_batch, k_batch, v_batch,
                output_f32[q_start:q_end], lse[q_start:q_end],
                q_start, kv_start, q_end, kv_end,
                sm_scale,
            )

        # Cast output to bfloat16 to match original API
        output = output_f32.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
