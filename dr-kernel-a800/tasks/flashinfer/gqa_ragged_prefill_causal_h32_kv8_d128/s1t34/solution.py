import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute logits[q, h, k] = sum_d q[q,h,d] * k_expanded[k,h,d]
# We reduce over D=128 using broadcasting; no runtime-dependent loops.
@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads, ceil(num_kv_tokens/BLOCK_K))
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    pid_k = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)      # [BLOCK_Q]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)      # [BLOCK_K]

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Load Q_tile: [BLOCK_Q, 128]
    Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + tl.arange(0, 128)[None, :] * Q_stride_d
    Q_tile = tl.load(Q_ptrs, mask=q_mask[:, None], other=0.0)  # [Q, 128], float32

    # Load K_exp_tile: [BLOCK_K, 128]
    K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + tl.arange(0, 128)[None, :] * K_EXP_stride_d
    K_tile = tl.load(K_ptrs, mask=k_mask[:, None], other=0.0)  # [K, 128], float32

    # Compute logits: [BLOCK_Q, BLOCK_K]
    # logits[q, k] = dot(Q_tile[q, :], K_tile[k, :]) over d
    # Broadcasted multiply and sum over d-axis
    logits = tl.sum(Q_tile[:, :, None] * K_tile[None, :, :], axis=1)  # [Q, K]

    # Apply scale
    logits = logits * SM_SCALE

    # Store
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, logits, mask=(q_mask[:, None] & k_mask[None, :]))


# Kernel 2: compute lse[q, h] = logsumexp(LOGITS[q, h, :]) / ln(2)
@triton.jit
def _lse_all_heads_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Initialize max and sum-exp
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Reduce over K in tiles
    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]
        # Update max
        max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))
        # Sum exp of shifted
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * ln2

    # Store
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


# Kernel 3: compute output[q, h, d] = sum_k softmax(LOGITS[q, h, k]) * V_expanded[k, h, d]
@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    q_mask = q_offsets < num_q_tokens

    # Load lse for this (q,h)
    LSE_ptrs = LSE + q_offsets * LSE.stride(0) + h * LSE.stride(1)
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Output accumulator [BLOCK_Q, 128]
    OUT_ptrs_base = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h
    out_acc = tl.zeros((BLOCK_Q, 128), dtype=tl.float32)

    # Reduce over K in tiles
    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]

        # Subtract lse for numerical stability
        vals = vals - lse_vals[:, None]  # [Q, K]

        exp_vals = tl.exp(vals)          # [Q, K]
        sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
        probs = exp_vals / sum_exp[:, None]  # [Q, K]

        # Multiply by V_exp[k,h,:] broadcast over d and sum over K
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            d_mask = d_idx < head_dim

            V_ptrs = V_EXP + k_offsets[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d  # [K, D]
            V_tile = tl.load(V_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D]

            # contrib: [Q, D] = sum_k probs[q,k] * V[k,d]
            contrib = tl.sum(probs[:, :, None] * V_tile[None, :, :], axis=1)  # [Q, D]

            # Add to accumulator
            out_acc += contrib

    # Store output
    OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + tl.arange(0, 128)[None, :] * OUT_stride_d  # [Q, 128]
    tl.store(OUT_ptrs, out_acc, mask=q_mask[:, None])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)

    def forward(self, q, k, v, qo_indptr, kv_indptr):
        # Ensure CUDA and dtype float32 for Triton
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Expand K/V by GQA ratio (num_qo_heads // num_kv_heads)
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1).contiguous()  # [num_kv_tokens, 32, 128]

        # Cast to float32 for Triton kernels
        q_f = q.to(torch.float32)
        kexp_f = k_expanded.to(torch.float32)
        vexp_f = v_expanded.to(torch.float32)

        # Prepare output and lse
        # We need to iterate over segments defined by indptrs
        device = q.device
        lse = torch.empty((q.shape[0], self.num_qo_heads), dtype=torch.float32, device=device)  # [total_q, 32]
        output = torch.empty((q.shape[0], self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)  # [total_q, 32, 128]

        # For each batch segment
        for b in range(0, qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Segment views for this batch
            q_seg = q_f[q_start:q_end]                        # [num_q_tokens, 32, 128]
            k_seg = kexp_f[kv_start:kv_end]                  # [num_kv_tokens, 32, 128]
            v_seg = vexp_f[kv_start:kv_end]                  # [num_kv_tokens, 32, 128]

            # Allocate intermediate
            logits_seg = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch 1) compute logits
            BLOCK_Q = 8   # tile over Q (num_q_tokens)
            BLOCK_K = 64  # tile over K (num_kv_tokens)
            grid1 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads, triton.cdiv(num_kv_tokens, BLOCK_K))
            _compute_logits_kernel[grid1](
                q_seg, k_seg, logits_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
                k_seg.stride(0), k_seg.stride(1), k_seg.stride(2),
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                SM_SCALE=self.sm_scale,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # Launch 2) lse per (q,h)
            grid2 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            ln2 = 1.0 / math.log(2.0)
            _lse_all_heads_kernel[grid2](
                logits_seg, lse[q_start:q_end],
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                lse[q_start:q_end].stride(0), lse[q_start:q_end].stride(1),
                ln2=ln2,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # Launch 3) softmax and output
            out_seg = output[q_start:q_end]  # [num_q_tokens, 32, 128]
            BLOCK_D = 16
            grid3 = (triton.cdiv(num_q_tokens, BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid3](
                logits_seg, v_seg, lse[q_start:q_end], out_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                v_seg.stride(0), v_seg.stride(1), v_seg.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

        # Cast output to bfloat16 as original returns
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
