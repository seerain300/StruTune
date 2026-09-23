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
    num_q_tokens, num_kv_tokens,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), ceil(num_kv_tokens/BLOCK_K), num_heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)          # [BLOCK_Q]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)          # [BLOCK_K]
    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits: shape [BLOCK_Q, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d with compile-time bound (HEAD_DIM = 128)
    for d0 in range(0, 128, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)                   # [BLOCK_D]
        d_mask = d_offsets < HEAD_DIM

        # Load Q[q, h, d] -> [BLOCK_Q, BLOCK_D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_offsets[None, :] * Q_stride_d
        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)  # [Q, D], fp32

        # Load K[k, h, d] -> [BLOCK_K, BLOCK_D]
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_offsets[None, :] * K_EXP_stride_d
        k_vals = tl.load(K_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D], fp32

        # Accumulate outer product: [Q, D] @ [K, D]^T -> [Q, K]
        # We do this per d-offset: acc += sum_d q_vals[:, d] * k_vals[:, d]
        # Because q_vals and k_vals are [Q,D] and [K,D], we reduce along D (axis=1) by summing elementwise products.
        # However, Triton supports broadcasting and tl.sum along an axis. We can compute:
        prod = q_vals[:, None, :] * k_vals[None, :, :]           # [Q, K, D]
        acc += tl.sum(prod, axis=2)                              # [Q, K]

    # Store acc to LOGITS[q, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & k_mask[None, :]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    LN2: tl.constexpr,
    delta: tl.constexpr,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Initialize max and sum_exp
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Reduce over K tiles, apply causal mask per q position
    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < num_kv_tokens

        allowed = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)  # [Q, K]
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]
        # Numerical stability
        m = tl.max(vals, axis=1)                                  # [Q]
        exp_vals = tl.exp(vals - m[:, None])                     # [Q, K]
        sum_exp += tl.sum(exp_vals, axis=1)                      # [Q]
        max_vals = tl.maximum(max_vals, m)                       # [Q]

    lse_vals = max_vals + tl.log(sum_exp) * LN2                 # [Q], per (q,h)
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse[q,h]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Compute output[q, h, d] = sum_k softmax(q,h,k) * V[k,h,d]
    for d0 in range(0, 128, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < HEAD_DIM

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_offsets[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for k0 in range(0, 128, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < num_kv_tokens

            allowed = k_offsets[None, :] < (q_offsets[:, None] + 1)  # [Q, K]
            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]

            # Subtract lse for numerical stability
            vals = vals - lse_vals[:, None]  # [Q, K]

            probs = tl.exp(vals) / tl.sum(tl.exp(vals), axis=1, keepdim=True)  # [Q, K]

            V_ptrs = V_EXP + k_offsets[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_offsets[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D]
            # out_row += sum_k probs[q,k] * v_vals[k,d]
            prod = probs[:, :, None] * v_vals[None, :, :]             # [Q, K, D]
            out_row += tl.sum(prod, axis=1)                           # [Q, D]

        # Store output
        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes and checks
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        device = q.device
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch segment
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Slicing
            q_batch = q[q_start:q_end]            # [num_q_tokens, 32, 128]
            k_batch = k[kv_start:kv_end]          # [num_kv_tokens, 8, 128]
            v_batch = v[kv_start:kv_end]          # [num_kv_tokens, 8, 128]

            # Expand k/v by GQA ratio
            k_expanded = k_batch.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]

            # Make tensors contiguous for Triton
            q_batch = q_batch.contiguous()
            k_expanded = k_expanded.contiguous()
            v_expanded = v_expanded.contiguous()

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_expanded.shape[0]
            delta = num_kv_tokens - num_q_tokens

            # Allocate intermediate
            logits = torch.empty((num_q_tokens, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits: grid depends on meta
            grid = lambda meta: (triton.cdiv(num_q_tokens, meta['BLOCK_Q']),
                                 triton.cdiv(num_kv_tokens, meta['BLOCK_K']),
                                 num_qo_heads)
            _compute_logits_kernel[grid](
                q_batch, k_expanded, logits,
                num_q_tokens, num_kv_tokens,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2),
                BLOCK_Q=1, BLOCK_K=64, BLOCK_D=16
            )

            # Launch Triton kernel to compute lse with causal mask
            grid_lse = lambda meta: (triton.cdiv(num_q_tokens, meta['BLOCK_Q']), num_qo_heads)
            _lse_masked_kernel[grid_lse](
                logits, lse[q_start:q_end],
                num_q_tokens, num_kv_tokens,
                logits.stride(0), logits.stride(1), logits.stride(2),
                lse.stride(0), lse.stride(1),
                LN2, delta,
                BLOCK_Q=1, BLOCK_K=64
            )

            # Launch Triton kernel to compute output: softmax(logits) @ v_expanded
            out_seg = output[q_start:q_end]
            grid_out = lambda meta: (triton.cdiv(num_q_tokens, meta['BLOCK_Q']), num_qo_heads)
            _softmax_output_kernel[grid_out](
                logits, v_expanded, lse[q_start:q_end], out_seg,
                num_q_tokens, num_kv_tokens,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=1, BLOCK_K=64, BLOCK_D=16
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
