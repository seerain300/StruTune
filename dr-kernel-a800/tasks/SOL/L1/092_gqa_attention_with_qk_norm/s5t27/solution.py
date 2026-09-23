import torch
import triton
import triton.language as tl

# Constants from the original code for this task
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
#   A: [M, K], B: [N, K], Bias: [N], C: [M, N]
@triton.jit
def linear_gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
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

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K
# Normalize along dim=-1 of a [B, S, H, D] tensor with weight [H, D]
# Output is [B, S, H, D] normalized: x / sqrt(mean(x^2) + eps) * weight
@triton.jit
def rms_norm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w_h, stride_w_d,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s = tmp // H
    h = tmp % H

    # Reduce over D to compute variance
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + RMS_EPS)

    # Apply normalization and weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + h * stride_w_h + offs_d * stride_w_d, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton rotate half: for each head, rotate the last half of D:
#    Input: X [B, S, H, D], Output: Y [B, S, H, D]
#    For D even, first_half = X[..., :D//2], last_half = X[..., D//2:], Y[..., :D//2] = -last_half, Y[..., D//2:] = first_half
@triton.jit
def rotate_half_kernel(
    X_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s = tmp // H
    h = tmp % H

    half = D // 2
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        q1 = x[:half]
        q2 = x[half:]
        y = tl.cat([-q2, q1], axis=0)  # first half is -q2, second half is q1
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs * stride_yd, y, mask=mask)

# 4) Triton GQA expand K/V from Hkv heads to Hq heads by repeating along groups NUM_KEY_VALUE_GROUPS
#    Inputs: K/V [B, S, Hkv, D], Output: Expanded [B, Hq, S, D]
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, S, Hkv, Hq, D, G,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * Hq
    if pid >= total:
        return
    b = pid // (S * Hq)
    tmp = pid % (S * Hq)
    s = tmp // Hq
    h_out = tmp % Hq
    h_in = h_out % Hkv  # map each output head to input head
    g = h_out // Hkv     # group index

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h_in * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h_out * stride_yh + offs_d * stride_yd, x, mask=mask)

# 5) Triton attention scores kernel: compute attn[b, h, i, j] = sum_k Q_norm[b,h,i,k] * K_exp[b,h,j,k] * scaling
#    Inputs: Q_norm [B, S, H, D], K_exp [B, S, H, D] (expanded 96 heads), Output: Attn [B, S, H, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ab, stride_as, stride_ah, stride_aj,
    scaling: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s_i = tmp // H
    h = tmp % H

    # Initialize attn[b, s_i, h, :] = 0
    attn_row = tl.zeros((S,), dtype=tl.float32)

    # Accumulate over k dimension (D)
    for k0 in range(0, D, BLOCK_D):
        offs_d = k0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D

        # Load Q[b, s_i, h, k]
        q = tl.load(Q_ptr + b * stride_qb + s_i * stride_qs + h * stride_qh + offs_d * stride_qd, mask=mask, other=0.0).to(tl.float32)  # [BD]
        for j in range(0, S):
            # Load K[b, j, h, k]
            k = tl.load(K_ptr + b * stride_kb + j * stride_ks + h * stride_kh + offs_d * stride_kd, mask=mask, other=0.0).to(tl.float32)  # [BD]
            prod = q * k  # elementwise multiply, [BD]
            attn_row[j] += tl.sum(prod, axis=0)  # sum over k tile

    # Apply scaling
    attn_row *= scaling

    # Store attn[b, s_i, h, :]
    a_ptrs = Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + tl.arange(0, S) * stride_aj
    tl.store(a_ptrs, attn_row, mask=tl.arange(0, S) < S)

# 6) Triton softmax per row (b, h, i) with causal mask (j > i): apply mask in-kernel
#    Inputs: Attn [B, S, H, S], Output: Soft [B, S, H, S]
@triton.jit
def softmax_rows_kernel(
    Attn_ptr, Soft_ptr,
    B, S, H,
    stride_ab, stride_as, stride_ah, stride_aj,
    stride_sb, stride_ss, stride_sh, stride_sj,
    BLOCK_S: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    tmp = pid % (S * H)
    s_i = tmp // H
    h = tmp % H

    # First pass: compute max for stability
    max_val = -float('inf')
    for j in range(0, S, BLOCK_S):
        offs = j + tl.arange(0, BLOCK_S)
        mask = offs < S
        attn = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs * stride_aj, mask=mask, other=-float('inf')).to(tl.float32)
        # causal mask: j <= s_i -> attn = -inf
        attn = tl.where(offs <= s_i, -float('inf'), attn)
        block_max = tl.max(attn, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum of exp
    sum_exp = 0.0
    for j in range(0, S, BLOCK_S):
        offs = j + tl.arange(0, BLOCK_S)
        mask = offs < S
        attn = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs * stride_aj, mask=mask, other=-float('inf')).to(tl.float32)
        attn = tl.where(offs <= s_i, -float('inf'), attn)
        expv = tl.exp(attn - max_val)
        sum_exp += tl.sum(expv, axis=0)

    # Third pass: write normalized values
    for j in range(0, S, BLOCK_S):
        offs = j + tl.arange(0, BLOCK_S)
        mask = offs < S
        attn = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs * stride_aj, mask=mask, other=-float('inf')).to(tl.float32)
        attn = tl.where(offs <= s_i, -float('inf'), attn)
        expv = tl.exp(attn - max_val)
        softv = expv / sum_exp
        tl.store(Soft_ptr + b * stride_sb + s_i * stride_ss + h * stride_sh + offs * stride_sj, softv, mask=mask)

# 7) Triton output projection (no bias): Output [B, S, Hq*D] = Attn_out [B, S, Hq*D] @ W_out [Hq*D, HvD]^T
#    Here we implement: Out[b, i, :] = sum_h Soft[b, i, h] * V_exp[b, h, i, :]
@triton.jit
def attn_output_projection_kernel(
    Soft_ptr, V_exp_ptr, Out_ptr,
    B, S, H, D, HvD,
    stride_sb, stride_ss, stride_sh,   # Soft strides
    stride_vb, stride_vs, stride_vh,   # V_exp strides
    stride_ob, stride_os, stride_od,   # Out strides
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S
    if pid >= total:
        return
    b = pid // S
    i = pid % S

    acc = tl.zeros((D,), dtype=tl.float32)
    for h in range(0, H):
        s_val = tl.load(Soft_ptr + b * stride_sb + i * stride_ss + h * stride_sh).to(tl.float32)
        for d0 in range(0, HvD, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask = offs_d < HvD
            v = tl.load(V_exp_ptr + b * stride_vb + h * stride_vh + i * stride_vs + offs_d * stride_vd,
                        mask=mask, other=0.0).to(tl.float32)
            acc += s_val * v
    # Store acc to Out[b,i,:]
    out_ptrs = Out_ptr + b * stride_ob + i * stride_os + tl.arange(0, D) * stride_od
    tl.store(out_ptrs, acc, mask=tl.arange(0, D) < D)

# 8) Triton output projection final: Output [B, S, Dq] = Out [B, S, Dq] @ o_proj_weight [Dq, HvD]^T (no bias)
@triton.jit
def output_projection_kernel(
    Out_ptr, W_ptr, Final_ptr,
    B, S, Dq, HvD,
    stride_ob, stride_os, stride_od,
    stride_wb, stride_wk, stride_wn,
    stride_fb, stride_fs, stride_fd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows over S
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols over Dq
    offs_k = tl.arange(0, BLOCK_K)                   # reduction over HvD

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, HvD, BLOCK_K):
        a_ptrs = Out_ptr + offs_m[:, None] * stride_ob + (k + offs_k)[None, :] * stride_od  # [BM, BK]
        b_ptrs = W_ptr + offs_n[None, :] * stride_wn + (k + offs_k)[:, None] * stride_wk     # [BK, BN]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < S) & ((k + offs_k)[None, :] < HvD), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < Dq) & ((k + offs_k)[:, None] < HvD), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = Final_ptr + offs_m[:, None] * stride_fb + offs_n[None, :] * stride_fd
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < S) & (offs_n[None, :] < Dq))

# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # Ensure CUDA and float32 for numeric stability
        assert hidden_states.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = torch.float32  # original code uses float32

        B, S, K_in = hidden_states.shape  # hidden_states: [B, S, 3*128*8] = [B, S, 3072]

        # 1) Linear projections: Q, K, V using Triton GEMM + bias
        Q = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        K = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)
        V = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)

        grid_q = (triton.cdiv(B * S, 64), triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, 128))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B * S, NUM_ATTENTION_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0) if q_proj_bias is not None else 0,
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        grid_k = (triton.cdiv(B * S, 64), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, 128))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B * S, NUM_KEY_VALUE_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0) if k_proj_bias is not None else 0,
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        grid_v = (triton.cdiv(B * S, 64), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, 128))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B * S, NUM_KEY_VALUE_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0) if v_proj_bias is not None else 0,
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, NUM_ATTENTION_HEADS, HEAD_DIM)    # [B, S, 96, 128]
        K_heads = K.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]
        V_heads = V.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]

        # 3) RMSNorm per head on Q and K (over last dim = 128)
        Q_norm = torch.empty_like(Q_heads, dtype=dtype)
        K_norm = torch.empty_like(K_heads, dtype=dtype)

        grid_rms_q = (B * S * NUM_ATTENTION_HEADS,)
        rms_norm_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0), q_norm_weight.stride(1),
            BLOCK_D=64
        )

        grid_rms_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rms_norm_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0), k_norm_weight.stride(1),
            BLOCK_D=64
        )

        # 4) Rotate last half for Q and K in Triton
        Q_rot = torch.empty_like(Q_norm, dtype=dtype)
        K_rot = torch.empty_like(K_norm, dtype=dtype)

        grid_rot_q = (B * S * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rot_q](
            Q_norm, Q_rot,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=64
        )

        grid_rot_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rot_k](
            K_norm, K_rot,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=64
        )

        # 5) GQA expand K and V to 96 heads (repeat along groups NUM_KEY_VALUE_GROUPS=12)
        K_exp = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=dtype)
        V_exp = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=dtype)

        grid_gqa_k = (B * S * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_gqa_k](
            K_rot, K_exp,
            B, S, NUM_KEY_VALUE_HEADS, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            BLOCK_D=64
        )

        grid_gqa_v = (B * S * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_gqa_v](
            V_heads, V_exp,
            B, S, NUM_KEY_VALUE_HEADS, NUM_ATTENTION_HEADS, HEAD_DIM, NUM_KEY_VALUE_GROUPS,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            BLOCK_D=64
        )

        # 6) Compute attention scores matrix Attn [B, S, 96, S] using Triton
        Attn = torch.empty((B, S, NUM_ATTENTION_HEADS, S), device=device, dtype=dtype)

        grid_attn = (B * S * NUM_ATTENTION_HEADS,)
        attn_scores_kernel[grid_attn](
            Q_rot, K_exp, Attn,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            scaling=SCALING,
            BLOCK_D=64
        )

        # 7) Softmax along j (seq position) per (b, h, i) with causal mask (triangular mask: j > i)
        Soft = torch.empty_like(Attn, dtype=dtype)

        grid_softmax = (B * S * NUM_ATTENTION_HEADS,)
        softmax_rows_kernel[grid_softmax](
            Attn, Soft,
            B, S, NUM_ATTENTION_HEADS,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            BLOCK_S=128
        )

        # 8) Compute attention output per (b, i): Out[b, i, :] = sum_h Soft[b, i, h] * V_exp[b, h, i, :]
        attn_output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)

        grid_proj_out = (B * S,)
        attn_output_projection_kernel[grid_proj_out](
            Soft, V_exp, attn_output,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            BLOCK_D=64
        )

        # 9) Final output projection: [B, S, Hq*D] @ o_proj_weight [Hq*D, HvD]^T (no bias)
        output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        grid_final = (triton.cdiv(B * S, 64), triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, 128))
        output_projection_kernel[grid_final](
            attn_output, o_proj_weight, output,
            B, S, NUM_ATTENTION_HEADS * HEAD_DIM, NUM_KEY_VALUE_HEADS * HEAD_DIM,
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1), o_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
