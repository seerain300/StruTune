import torch
import triton
import triton.language as tl

# Constants consistent with the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)

# 1) Triton GEMM with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
@triton.jit
def linear_gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + (k + offs_k)[:, None] * stride_bk)  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K), other=0.0)

        acc += tl.dot(a, b)  # [BM, BN]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BN]
    acc += bias[None, :]  # broadcast over rows

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Triton RMSNorm per head over last dim (HEAD_DIM=128): y[b, s, h, :] = x[b, s, h, :] * rsqrt(mean(x^2) + eps)
@triton.jit
def rmsnorm_lastdim_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,
    BLOCK_D: tl.constexpr,
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s = tmp // H
    h = tmp % H

    # Reduction over D
    sum_sq = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=(offs_d < D), other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 1e-6)  # eps = 1e-6 as in original code

    # Scale and apply weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=(offs_d < D), other=0.0)
        w = tl.load(Weight_ptr + h * stride_w)
        y = x * inv_rms * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=(offs_d < D))


# 3) Triton kernel to rotate last half: Q_norm or K_norm -> [-q2, q1]
# Input X: [B, S, H, D] float32, Output Y: same shape, in-place rotation only for half dims.
@triton.jit
def rotate_last_half_inplace_kernel(
    X_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr,
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s = tmp // H
    h = tmp % H

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D

        # Load x
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)

        # Split into two halves
        half = D // 2
        q1 = x[:half]
        q2 = x[half:]

        # Form rotated: [-q2, q1]
        rotated = tl.zeros((D,), dtype=tl.float32)
        rotated[:half] = -q2
        rotated[half:] = q1

        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, rotated, mask=mask)


# 4) Triton GQA expand: K or V [B, S, Hkv, D] -> [B, S, Hq, D] by repeating along groups
# We assume Hq=NUM_ATTENTION_HEADS and Hkv=NUM_KEY_VALUE_HEADS, groups=NUM_KEY_VALUE_GROUPS
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, S, Hkv, Hq, D, G,  # G = NUM_KEY_VALUE_GROUPS
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr,
):
    total = B * S * Hq
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * Hq)
    tmp = pid % (S * Hq)
    hq = tmp  # target head index
    s = 0  # we iterate over S and Hkv inside
    # For each source head hkv, write to all groups g
    for s_idx in range(0, S):
        for hkv in range(0, Hkv):
            # Compute source vector index
            for g_idx in range(0, G):
                h_src = hkv * G + g_idx
                # Copy X[b, s_idx, h_src, :] into Y[b, s_idx, hq, :]
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask = offs_d < D
                    x = tl.load(
                        X_ptr + b * stride_xb + s_idx * stride_xs + h_src * stride_xh + offs_d * stride_xd,
                        mask=mask, other=0.0
                    )
                    tl.store(
                        Y_ptr + b * stride_yb + s_idx * stride_ys + hq * stride_yh + offs_d * stride_yd,
                        x, mask=mask
                    )


# 5) Triton attention scores: compute attn_scores[b, h, i, j] = sum_k Q_norm[b,h,i,k] * K_expanded[b,h,j,k] * SCALING
# Output is [B, S, Hq, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_exp_ptr, Out_ptr,
    B, S, Hq, D,  # K_exp shape: [B, S, Hq, S, D], Out: [B, S, Hq, S]
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_keb, stride_kes, stride_keh, stride_ked, stride_kei,
    stride_ob, stride_os, stride_oh, stride_os2,
    BLOCK_D: tl.constexpr,
):
    total = B * S * Hq
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * Hq)
    tmp = pid % (S * Hq)
    h = tmp
    for i in range(0, S):
        # Accumulate scores for all j
        scores = tl.zeros((S,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load Q[b, i, h, offs_d]
            q = tl.load(
                Q_ptr + b * stride_qb + i * stride_qs + h * stride_qh + offs_d * stride_qd,
                mask=mask_d, other=0.0
            )

            # For each j, compute dot with K_exp[b, j, h, i, offs_d]
            for j in range(0, S):
                k = tl.load(
                    K_exp_ptr + b * stride_keb + j * stride_kes + h * stride_keh + i * stride_kei + offs_d * stride_ked,
                    mask=mask_d, other=0.0
                )
                scores[j] += tl.sum(q * k, axis=0)

        # Apply scaling
        scores *= SCALING

        # Store scores[b, i, h, :]
        out_ptrs = Out_ptr + b * stride_ob + i * stride_os + h * stride_oh + tl.arange(0, S) * stride_os2
        # Create mask for j in [0..S)
        j_vec = tl.arange(0, S)
        tl.store(out_ptrs, scores, mask=(j_vec < S))


# 6) Triton softmax rows with causal mask: for each (b, h) row over i dimension
# We implement per row softmax over last dimension S with Softmax = exp(x)/sum(exp(x)), where x is attn_scores.
@triton.jit
def softmax_rows_causal_kernel(
    In_ptr, Out_ptr,
    B, S, H,
    stride_ib, stride_is, stride_ih,
    stride_ob, stride_os, stride_oh,
    BLOCK_S: tl.constexpr,
):
    total = B * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // H
    h = pid % H

    # Compute max over i (row), then exp and sum, then store normalized
    row_max = -1e30
    for i in range(0, S):
        val = tl.load(In_ptr + b * stride_ib + i * stride_is + h * stride_ih)
        if val > row_max:
            row_max = val

    row_sum = 0.0
    for i in range(0, S):
        val = tl.load(In_ptr + b * stride_ib + i * stride_is + h * stride_ih)
        e = tl.exp(val - row_max)
        row_sum += e

    for i in range(0, S):
        val = tl.load(In_ptr + b * stride_ib + i * stride_is + h * stride_ih)
        out_val = tl.exp(val - row_max) / row_sum
        tl.store(Out_ptr + b * stride_ob + i * stride_os + h * stride_oh, out_val)


# 7) Triton output projection: Out [B, S, Hq*D] = Out [B, S, 12288] @ o_proj_weight^T (no bias)
# We implement a simple GEMM: for each (b, s), vector out_vec [Hq*D], multiply by o_proj_weight [D_out, Dq], accumulate to Out [B, S, D_out]
@triton.jit
def output_projection_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, S, Dq, Dout,
    stride_xb, stride_xs, stride_xd,
    stride_wm, stride_wn,  # W is [Dout, Dq]
    stride_ob, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    total = B * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // S
    s = pid % S

    acc = tl.zeros((Dout,), dtype=tl.float32)
    for d0 in range(0, Dq, BLOCK_D):
        offs_dq = d0 + tl.arange(0, BLOCK_D)
        mask_dq = offs_dq < Dq
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + offs_dq * stride_xd, mask=mask_dq, other=0.0)
        for m0 in range(0, Dout, BLOCK_D):
            offs_m = m0 + tl.arange(0, BLOCK_D)
            mask_m = offs_m < Dout
            w = tl.load(W_ptr + offs_m[:, None] * stride_wm + offs_dq[None, :] * stride_wn,
                        mask=mask_m[:, None] & mask_dq[None, :], other=0.0)
            acc[offs_m] += tl.sum(w * x[None, :], axis=1)

    tl.store(Out_ptr + b * stride_ob + s * stride_os + tl.arange(0, Dout) * stride_od, acc, mask=(tl.arange(0, Dout) < Dout))

# Entry point class
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # All tensors must be on CUDA for Triton
        assert hidden_states.is_cuda, "Input tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype

        B, S, Dq_in = hidden_states.shape  # hidden_states: [B, S, 12288]
        Dq = Dq_in  # 12288
        D = HEAD_DIM  # 128
        Hq = NUM_ATTENTION_HEADS  # 96
        Hkv = NUM_KEY_VALUE_HEADS  # 8
        G = NUM_KEY_VALUE_GROUPS  # 12

        # 1) Linear projections using Triton GEMM + bias: Q, K, V
        # Shapes:
        # hidden_states: [B, S, Dq_in]
        # q_proj_weight: [Dq_in, Dq_out] where Dq_out = Hq * D = 12288
        # k_proj_weight: [Dq_in, Hkv*D] where Hkv*D = 1024
        # v_proj_weight: [Dq_in, Hkv*D] where Hkv*D = 1024

        # Allocate outputs
        Q = torch.empty((B, S, Hq * D), device=device, dtype=torch.float32)
        K_raw = torch.empty((B, S, Hkv * D), device=device, dtype=torch.float32)
        V_raw = torch.empty((B, S, Hkv * D), device=device, dtype=torch.float32)

        # Launch linear kernels
        # BLOCK_M=64, BLOCK_N=64, BLOCK_K=64 are good defaults for 12288x12288 etc.
        grid_q = (B, S, Hq)
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, Dq_in, Hq * D,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            64, 64, 64,
        )

        grid_k = (B, S, Hkv)
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K_raw,
            B, S, Dq_in, Hkv * D,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K_raw.stride(0), K_raw.stride(1),
            64, 64, 64,
        )

        grid_v = (B, S, Hkv)
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V_raw,
            B, S, Dq_in, Hkv * D,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V_raw.stride(0), V_raw.stride(1),
            64, 64, 64,
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, Hq, D)   # [B, S, 96, 128]
        K_heads = K_raw.view(B, S, Hkv, D)  # [B, S, 8, 128]
        V_heads = V_raw.view(B, S, Hkv, D)  # [B, S, 8, 128]

        # 3) RMSNorm per head over last dim (128) for Q and K
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms_q = (B * S * Hq,)
        rmsnorm_lastdim_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S, Hq, D,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0),
            BLOCK_D=128,
        )

        grid_rms_k = (B * S * Hkv,)
        rmsnorm_lastdim_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S, Hkv, D,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0),
            BLOCK_D=128,
        )

        # 4) Rotate last half for Q and K: rotate(-q2, q1)
        # Implement rotation in-place into new tensors
        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32)

        grid_rotate_q = (B * S * Hq,)
        rotate_last_half_inplace_kernel[grid_rotate_q](
            Q_norm, Q_rot,
            B, S, Hq, D,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=128,
        )

        grid_rotate_k = (B * S * Hkv,)
        rotate_last_half_inplace_kernel[grid_rotate_k](
            K_norm, K_rot,
            B, S, Hkv, D,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=128,
        )

        # 5) Grouped-Query Attention: expand K_rot and V_heads from Hkv to Hq using groups=G=12
        K_exp = torch.empty((B, S, Hq, S, D), device=device, dtype=torch.float32)
        V_exp = torch.empty((B, S, Hq, S, D), device=device, dtype=torch.float32)

        grid_gqa_k = (B * S * Hq,)
        gqa_expand_kernel[grid_gqa_k](
            K_rot, K_exp,
            B, S, Hkv, Hq, D, G,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            BLOCK_D=128,
        )

        grid_gqa_v = (B * S * Hq,)
        gqa_expand_kernel[grid_gqa_v](
            V_heads, V_exp,
            B, S, Hkv, Hq, D, G,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            BLOCK_D=128,
        )

        # 6) Compute attention scores: [B, S, Hq, S]
        attn_scores = torch.empty((B, S, Hq, S), device=device, dtype=torch.float32)

        grid_attn = (B * S * Hq,)
        attn_scores_kernel[grid_attn](
            Q_rot, K_exp, attn_scores,
            B, S, Hq, D,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3), K_exp.stride(4),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            BLOCK_D=128,
        )

        # 7) Softmax along the last dimension S for each (b, h) row with causal mask (triangular: j > i)
        attn_probs = torch.empty_like(attn_scores, dtype=torch.float32)

        grid_softmax = (B * Hq,)
        softmax_rows_causal_kernel[grid_softmax](
            attn_scores, attn_probs,
            B, S, Hq,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2),
            BLOCK_S=128,
        )

        # 8) Compute attention output per (b, s, h): Out[b, s, h, :] = sum_j attn[b, h, s, j] * V_exp[b, h, j, :]
        attn_output = torch.empty((B, S, Hq * D), device=device, dtype=torch.float32)

        # Implement per-head accumulation via Triton kernel. We need vectorized access to V_exp[b, h, j, :] for j in [0..S-1].
        # We can write a kernel that for each (b, s, h), loads each j, multiplies by attn_probs[b, h, s, j], and accumulates into attn_output[b, s, h*128:(h+1)*128].
        @triton.jit
        def attn_output_kernel(
            Probs_ptr, V_exp_ptr, Out_ptr,
            B, S, Hq, D,
            stride_pb, stride_ps, stride_ph,
            stride_vb, stride_vs, stride_vh, stride_vi, stride_vd,
            stride_ob, stride_os, stride_oh, stride_od,
            BLOCK_S: tl.constexpr,
        ):
            total = B * S * Hq
            pid = tl.program_id(0)
            if pid >= total:
                return
            b = pid // (S * Hq)
            tmp = pid % (S * Hq)
            s = tmp // Hq
            h = tmp % Hq

            out_base = Out_ptr + b * stride_ob + s * stride_os + h * D * stride_oh  # h*128 starts at h * D
            for d0 in range(0, D, BLOCK_S):
                offs_d = d0 + tl.arange(0, BLOCK_S)
                mask_d = offs_d < D
                acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
                # For each j, read probs[b, h, s, j] and V_exp[b, h, j, offs_d], multiply, and accumulate
                for j in range(0, S):
                    prob = tl.load(Probs_ptr + b * stride_pb + s * stride_ps + h * stride_ph, mask=True, other=0.0)  # read scalar
                    v = tl.load(
                        V_exp_ptr + b * stride_vb + s * stride_vs + h * stride_vh + j * stride_vi + offs_d * stride_vd,
                        mask=mask_d, other=0.0
                    )
                    acc += prob * v
                tl.store(out_base + offs_d * stride_od, acc, mask=mask_d)

        grid_out = (B * S * Hq,)
        attn_output_kernel[grid_out](
            attn_probs, V_exp, attn_output,
            B, S, Hq, D,
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3), V_exp.stride(4),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
            BLOCK_S=64,
        )

        # Flatten to [B, S, Hq*D]
        attn_output_flat = attn_output.view(B, S, Hq * D)

        # 9) Final output projection using Triton (no bias)
        output = torch.empty((B, S, o_proj_weight.shape[0]), device=device, dtype=torch.float32)

        # o_proj_weight shape is [D_out, Dq], where Dq = Hq * D = 12288, D_out = o_proj_weight.shape[0]
        Dout = o_proj_weight.shape[0]
        grid_proj = (B * S,)
        output_projection_kernel[grid_proj](
            attn_output_flat, o_proj_weight, output,
            B, S, Hq * D, Dout,
            attn_output_flat.stride(0), attn_output_flat.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_D=128,
        )

        return output


def run(*args):
    return ModelNew()(*args)
