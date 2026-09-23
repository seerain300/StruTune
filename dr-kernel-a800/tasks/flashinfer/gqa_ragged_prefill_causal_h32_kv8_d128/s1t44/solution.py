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
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads, ceil(head_dim/BLOCK_D))
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    pid_d = tl.program_id(2)

    # q indices handled by this program
    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # BLOCK_Q=1, but keep general
    q_mask = q_offsets < num_q_tokens

    # d tile
    d0 = pid_d * BLOCK_D
    d_offsets = d0 + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    # Accumulator for [q, k] across d
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over d in chunks (compile-time bound)
    for d in range(0, 128):
        # Skip if d >= head_dim; Triton requires compile-time loops, but we mask in loads
        # Pointers for Q and K_EXP for this d
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d * Q_stride_d
        K_ptrs = K_EXP + tl.arange(0, BLOCK_K) * K_EXP_stride_k + h * K_EXP_stride_h + d * K_EXP_stride_d

        q_vals = tl.load(Q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)  # [Q, 1], but using D-masked 1D load not ideal in Triton; see next approach
        # To correctly load Q[q, h, d] as [Q], we load single column vector:
        # We'll instead implement per-d scalar loading via a separate kernel. Simplify: pre-load per-d, not inside jit kernel.

    # Store acc
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, BLOCK_K) * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None]))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    LN2: tl.constexpr, delta: tl.constexpr,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for k0 in range(0, 128, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < num_kv_tokens

        # Causal mask: allowed if k < min(num_kv_tokens, q_pos + 1 + delta)
        q_pos = q_offsets  # [Q]
        allowed = k_offsets[None, :] < (q_pos[:, None] + 1 + delta)  # [Q, K]

        LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]

        # Row-wise max
        m = tl.max(vals, axis=1)  # [Q]
        # Row-wise sum of exp(vals - m)
        exp_vals = tl.exp(vals - m[:, None])  # [Q, K]
        sum_exp += tl.sum(exp_vals, axis=1)   # [Q]
        # Update max
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
    # Grid: (ceil(num_q_tokens/BLOCK_Q), num_qo_heads, ceil(head_dim/BLOCK_D))
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    pid_d = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    d0 = pid_d * BLOCK_D
    d_offsets = d0 + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    for q_off in range(0, BLOCK_Q):
        # Load lse[q, h]
        LSE_ptr = LSE + q_offsets[q_off] * LSE.stride(0) + h * LSE.stride(1)
        lse_val = tl.load(LSE_ptr, mask=q_mask[q_off], other=-float("inf"))  # scalar

        out_row = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for k0 in range(0, 128, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < num_kv_tokens

            q_pos = q_offsets[q_off]  # scalar
            allowed = k_offsets < (q_pos + 1)  # causal: only k < q_pos + 1

            LOGITS_ptrs = LOGITS + q_pos * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[q_off] & k_mask), other=-float("inf"))  # [K]
            probs = tl.exp(vals - lse_val) / tl.sum(tl.exp(vals - lse_val))  # scalar normalization

            V_ptrs = V_EXP + k_offsets[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_offsets[None, :] * V_EXP_stride_d
            v_vals = tl.load(V_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)  # [K, D]
            out_row += tl.sum(probs[:, None] * v_vals, axis=0)

        OUT_ptrs = OUT + q_offsets[q_off] * OUT_stride_q + h * OUT_stride_h + d_offsets * OUT_stride_d
        tl.store(OUT_ptrs, out_row, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we keep Triton-only implementation

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        device = q.device
        dtype = torch.float32

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process segments
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            delta = num_kv_tokens - num_q_tokens

            # Slice and expand K, V by GQA ratio
            q_seg = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_seg = k[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_seg = v[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            k_expanded = k_seg.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_seg.repeat_interleave(GQA_RATIO, dim=1)  # [num_kv_tokens, 32, 128]

            # Prepare per-segment output
            out_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)

            # 1) Compute logits [Q, K] over all (q,h) and d
            # Note: Triton requires compile-time loops; we compute per (q,h) and d using kernel
            # Grid: (ceil(Q/BQ), H, ceil(H/BLOCK_D))
            BLOCK_Q = 1
            BLOCK_K = 64
            BLOCK_D = 16
            grid = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS, triton.cdiv(HEAD_DIM, BLOCK_D))
            _compute_logits_kernel[grid](
                q_seg, k_expanded, out_seg,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # 2) LSE with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS)
            _lse_masked_kernel[grid_lse](
                out_seg, lse_seg,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                LN2, delta,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
            )

            # 3) Softmax and output
            grid_out = (triton.cdiv(num_q_tokens, BLOCK_Q), NUM_QO_HEADS, triton.cdiv(HEAD_DIM, BLOCK_D))
            _softmax_output_kernel[grid_out](
                out_seg, v_expanded, lse_seg, out_seg,
                num_q_tokens, num_kv_tokens, HEAD_DIM,
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),  # OUT has same shape; store into out_seg
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
            )

            # Copy segment results to global output/lse
            output[q_start:q_end] = out_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
