import torch
import math
import triton
import triton.language as tl

# Constants from original Model
HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # 4
LN2 = math.log(2.0)


@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), ceil(num_kv_tokens/BLOCK_K), heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator over d dimension (head_dim=128)
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d in chunks of BLOCK_D with compile-time bounds
    for d0 in range(0, 128, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_offsets < head_dim

        # Load Q[q, h, d] -> [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_offsets[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)  # [Q, D], fp32

        # Load K[k, h, d] -> [BLOCK_K, BLOCK_D]
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_offsets[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D], fp32

        # Accumulate outer product over D: acc += sum_d q_vals * k_vals^T
        prod = q_vals[:, None, :] * k_vals[None, :, :]  # [Q, K, D]
        acc += tl.sum(prod, axis=2)

    # Store acc to LOGITS[q, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Accumulator for max and sum
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Iterate K chunks up to head_dim with compile-time constant bound
    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]

        # Causal mask: allowed keys j < (q_pos + 1 + delta), delta = num_kv_tokens - num_q_tokens
        delta = num_kv_tokens - num_q_tokens
        allowed = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)

        # Apply mask
        vals = tl.where(allowed, vals, -float("inf"))

        # Reduce over K to get max and sum(exp(vals - max))
        m = tl.max(vals, axis=1)
        exp_vals = tl.exp(vals - m[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

        max_vals = tl.maximum(max_vals, m)

    lse_vals = max_vals + tl.log(sum_exp) * LN2  # per (q,h)
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Load lse for this (q,h)
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Compute output[q, h, d] across d tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [Q]
            delta = num_kv_tokens - num_q_tokens
            allowed = k_idx[None, :] < (q_pos[:, None] + 1 + delta)  # [Q, K]

            LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]

            # Subtract lse for numerical stability
            vals = vals - lse_vals[:, None]  # [Q, K]

            exp_vals = tl.exp(vals)          # [Q, K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]

            # Accumulate: out_row += sum_k probs[q,k] * v_vals[k,d]
            prod = probs[:, :, None] * v_vals[None, :, :]  # [Q, K, D]
            out_row += tl.sum(prod, axis=1)                # reduce over K -> [Q, D]

        # Store out_row to OUT[q, h, d]
        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        sm_scale: float (unused; kept for signature compatibility)
        Returns (output: [total_q, 32, 128], lse: [total_q, 32]), both float32
        """
        device = q.device
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        assert q.shape[1] == NUM_QO_HEADS and k.shape[1] == NUM_KV_HEADS and k.shape[2] == HEAD_DIM
        assert qo_indptr is not None and kv_indptr is not None
        len_indptr = qo_indptr.shape[0]
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        assert total_q == q.shape[0] and total_kv == k.shape[0]

        # Output buffers
        output = torch.empty((total_q, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, NUM_QO_HEADS), dtype=torch.float32, device=device)

        # Iterate segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            # Slice and expand (compute in fp32)
            q_seg = q[q_start:q_end].contiguous().to(torch.float32)  # [num_q_tokens, 32, 128]
            k_seg = k[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]
            v_seg = v[kv_start:kv_end].contiguous().to(torch.float32)  # [num_kv_tokens, 8, 128]

            k_exp = k_seg.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]
            v_exp = v_seg.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate LOGITS for this segment [num_q_tokens, num_kv_tokens]
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch compute logits
            BLOCK_Q = 1
            BLOCK_K = 64
            BLOCK_D = 16
            grid = (triton.cdiv(num_q_tokens, BLOCK_Q), triton.cdiv(num_kv_tokens, BLOCK_K), NUM_QO_HEADS)
            _compute_logits_kernel[grid](
                q_seg, k_exp, logits,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
                k_exp.stride(0), k_exp.stride(1), k_exp.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(1),  # LOGITS_stride_k is 1
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # Compute lse per (q,h)
            lse_seg = torch.empty((num_q_tokens, NUM_QO_HEADS), dtype=torch.float32, device=device)
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS)
            _lse_masked_kernel[grid_lse](
                logits, lse_seg,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                logits.stride(0), logits.stride(1), logits.stride(1),  # LOGITS_stride_k is 1
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=BLOCK_Q, BLOCK_K=64
            )
            lse[q_start:q_end] = lse_seg

            # Compute output
            out_seg = torch.empty((num_q_tokens, NUM_QO_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS)
            _softmax_output_kernel[grid_out](
                logits, v_exp, lse_seg, out_seg,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                logits.stride(0), logits.stride(1), logits.stride(1),  # LOGITS_stride_k is 1
                v_exp.stride(0), v_exp.stride(1), v_exp.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=64, BLOCK_D=16
            )
            output[q_start:q_end] = out_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
