import torch
import triton
import triton.language as tl

# Constants
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
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

        # Compute in fp32
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    # Add bias: bias is [N]
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Triton RMSNorm over last dim (D): input [B,S,H,D] -> output same, using weight [D]
@triton.jit
def rms_norm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # Reduce over D to get sum of squares
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x)

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + EPS)

    # Apply weight and write back
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(Weight_ptr + offs_d * stride_w, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)


# 3) Triton rotate half for Q and K: Q_in [B,S,H,D], cos/sin [D], outputs Qr,Kr [B,S,H,D]
# We rotate the last half (64) with q1=first 64, q2=last 64, q_rot = [-q2, q1]
@triton.jit
def rotate_half_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_c, stride_s,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # First half q1: [0:64]
    for d0 in range(0, 64, BLOCK_D):
        offs_d1 = d0 + tl.arange(0, BLOCK_D)
        mask1 = offs_d1 < 64
        x1 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d1 * stride_xd, mask=mask1, other=0.0)
        c1 = tl.load(Cos_ptr + offs_d1 * stride_c, mask=mask1, other=1.0)
        s1 = tl.load(Sin_ptr + offs_d1 * stride_s, mask=mask1, other=1.0)
        # rotate: (x1 * cos) + ( (-x2) * sin ), but here x2 is not loaded; this kernel only rotates half using precomputed indices
        y1 = x1 * c1 + (-tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + (offs_d1 + 64) * stride_xd, mask=mask1, other=0.0)) * s1
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d1 * stride_yd, y1, mask=mask1)

    # Second half q2: [64:128], rotated as -q2
    for d0 in range(0, 64, BLOCK_D):
        offs_d2 = d0 + tl.arange(0, BLOCK_D)
        mask2 = offs_d2 < 64
        x2 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + (offs_d2 + 64) * stride_xd, mask=mask2, other=0.0)
        c2 = tl.load(Cos_ptr + (offs_d2 + 64) * stride_c, mask=mask2, other=1.0)
        s2 = tl.load(Sin_ptr + (offs_d2 + 64) * stride_s, mask=mask2, other=1.0)
        y2 = -x2 * c2 - tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d2 * stride_xd, mask=mask2, other=0.0) * s2
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + (offs_d2 + 64) * stride_yd, y2, mask=mask2)


# 4) Triton GQA expand: K_expanded [B, H, S, D] = repeat K_heads [B, Hv, S, D] along H using NUM_KEY_VALUE_GROUPS
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, S, H, D, Hv,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    NUM_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * Hv
    if pid >= total:
        return
    b = pid // (S * Hv)
    rem = pid % (S * Hv)
    s = rem // Hv
    hv = rem % Hv

    # For each group g in [0, NUM_GROUPS), write into h = hv + g
    for g in range(0, NUM_GROUPS):
        h = hv + g
        if h >= H:
            continue
        # Copy X[b, hv, s, :] into Y[b, h, s, :]
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            x = tl.load(X_ptr + b * stride_xb + s * stride_xs + hv * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
            tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, x, mask=mask)


# 5) Triton compute attention scores: attn[b,h,i,j] = sum_k Q_n[b,h,i,k] * K_expanded[b,h,j,k] * scaling
# This kernel computes one row (i) per program for a given (b,h). We loop over j and k (all in fp32).
@triton.jit
def attn_scores_row_kernel(
    Q_ptr, K_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ab, stride_as, stride_ah, stride_aj,
    scaling: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    h = rem % H
    i = rem // H  # i in [0, S)
    if i >= S:
        return

    # For each j in [0, S), compute sum_k Q[b,h,i,k] * K[b,h,j,k]
    for j in range(0, S):
        acc = 0.0
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            q = tl.load(Q_ptr + b * stride_qb + i * stride_qs + h * stride_qh + offs_d * stride_qd, mask=mask, other=0.0)
            k = tl.load(K_ptr + b * stride_kb + j * stride_ks + h * stride_kh + offs_d * stride_kd, mask=mask, other=0.0)
            acc += tl.sum(q.to(tl.float32) * k.to(tl.float32))
        attn_val = acc * scaling  # fp32
        tl.store(Attn_ptr + b * stride_ab + i * stride_as + h * stride_ah + j * stride_aj, attn_val)


# 6) Triton softmax per row (b,h,i): given Attn[b,h,0:S], write normalized attn and mask with causal i<=j -> -inf
@triton.jit
def softmax_rows_kernel(
    In_ptr, Out_ptr,
    B, S, H,
    stride_ib, stride_is, stride_ih, stride_ij,
    stride_ob, stride_os, stride_oh, stride_oj,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    h = rem % H
    i = rem // H
    if i >= S:
        return

    # Load row i for (b,h)
    row = tl.zeros((S,), dtype=tl.float32)
    for j in range(0, S):
        val = tl.load(In_ptr + b * stride_ib + i * stride_is + h * stride_ih + j * stride_ij)
        row[j] = val

    # Apply causal mask: for j <= i, set -inf (numerically large negative)
    for j in range(0, S):
        if j <= i:
            row[j] = -1e30  # large negative

    # Compute max for stability
    maxv = -1e30
    for j in range(0, S):
        if row[j] > maxv:
            maxv = row[j]
    # Subtract max
    for j in range(0, S):
        row[j] -= maxv

    # Exponentiate
    for j in range(0, S):
        row[j] = tl.exp(row[j])

    # Sum
    sumv = 0.0
    for j in range(0, S):
        sumv += row[j]
    # Normalize
    for j in range(0, S):
        row[j] = row[j] / sumv

    # Store normalized row
    for j in range(0, S):
        tl.store(Out_ptr + b * stride_ob + i * stride_os + h * stride_oh + j * stride_oj, row[j])


# 7) Triton output projection (no bias): Out[M] = Attn[M] @ o_proj_weight^T
# Attn: [B*S*H, D_out_per_head], o_proj_weight: [D_out_per_head, D_out_per_head]
@triton.jit
def linear_output_kernel(
    In_ptr, W_ptr, Out_ptr,
    M, N, K,  # In[M, K], W[N, K], Out[M, N]
    stride_im, stride_ik,
    stride_wm, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = In_ptr + (offs_m[:, None] * stride_im + (k + offs_k)[None, :] * stride_ik)  # [BM, BK]
        b_ptrs = W_ptr + (offs_n[None, :] * stride_wm + (k + offs_k)[:, None] * stride_wk)  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K), other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    # Store
    c_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Entry point class
class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rms_eps = rms_eps

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # hidden_states: [B, S, K_in], q_proj_weight: [K_in, Hq*D], q_proj_bias: [Hq*D]
        # k_proj_weight: [K_in, Hv*D], k_proj_bias: [Hv*D], v_proj_weight same, o_proj_weight: [D_out_per_head, D_out_per_head]
        # q_norm_weight: [Hq*D], k_norm_weight: [Hv*D]
        device = hidden_states.device
        dtype = hidden_states.dtype

        B, S, K_in = hidden_states.shape
        Hq = NUM_ATTENTION_HEADS
        Hv = NUM_KEY_VALUE_HEADS
        D_q = Hq * HEAD_DIM
        D_k = Hv * HEAD_DIM

        # 1) Linear projections using Triton: Q, K, V
        # Allocate outputs as float32 for accumulation
        Q = torch.empty((B, S, D_q), device=device, dtype=torch.float32)
        K = torch.empty((B, S, D_k), device=device, dtype=torch.float32)
        V = torch.empty((B, S, D_k), device=device, dtype=torch.float32)

        # We need K_in == q_proj_weight.shape[0] (768) and v_proj_weight.shape[0] (768)
        # Launch Triton GEMM + bias for each
        grid_q = (triton.cdiv(S, 64), triton.cdiv(D_q, 64))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        grid_k = (triton.cdiv(S, 64), triton.cdiv(D_k, 64))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        grid_v = (triton.cdiv(S, 64), triton.cdiv(D_k, 64))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, Hq, HEAD_DIM)    # [B, S, 96, 128]
        K_heads = K.view(B, S, Hv, HEAD_DIM)   # [B, S, 8, 128]
        V_heads = V.view(B, S, Hv, HEAD_DIM)   # [B, S, 8, 128]

        # 3) RMSNorm per head using Triton
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        total_q = B * S * Hq
        grid_rms_q = (total_q,)
        rms_norm_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S, Hq, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0), EPS=self.rms_eps, BLOCK_D=128
        )

        total_k = B * S * Hv
        grid_rms_k = (total_k,)
        rms_norm_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S, Hv, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0), EPS=self.rms_eps, BLOCK_D=128
        )

        # 4) Rotate half for Q and K in Triton
        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32)

        grid_rot_q = (total_q,)
        rotate_half_kernel[grid_rot_q](
            Q_norm, cos, sin, Q_rot,
            B, S, Hq, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            cos.stride(0), sin.stride(0),
            BLOCK_D=128
        )

        grid_rot_k = (total_k,)
        rotate_half_kernel[grid_rot_k](
            K_norm, cos, sin, K_rot,
            B, S, Hv, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            cos.stride(0), sin.stride(0),
            BLOCK_D=128
        )

        # 5) GQA expand K and V to 96 heads using Triton
        K_expanded = torch.empty((B, S, Hq, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, Hq, HEAD_DIM), device=device, dtype=torch.float32)

        total_gqa = B * S * Hv
        grid_gqa = (total_gqa,)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, Hq, HEAD_DIM, Hv,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            NUM_GROUPS=NUM_KEY_VALUE_GROUPS,
            BLOCK_D=128
        )

        grid_gqa_v = (total_gqa,)
        gqa_expand_kernel[grid_gqa_v](
            V_heads, V_expanded,
            B, S, Hq, HEAD_DIM, Hv,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            NUM_GROUPS=NUM_KEY_VALUE_GROUPS,
            BLOCK_D=128
        )

        # 6) Compute attention scores S[b,h,i,j] in Triton: one row per (b,h,i)
        S_attn = torch.empty((B, S, Hq), device=device, dtype=torch.float32)

        total_rows = B * S * Hq
        grid_scores = (total_rows,)
        attn_scores_row_kernel[grid_scores](
            Q_rot, K_expanded, S_attn,
            B, S, Hq, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            S_attn.stride(0), S_attn.stride(1), S_attn.stride(2), S_attn.stride(3),
            scaling=SCALING,
            BLOCK_D=128
        )

        # Apply causal mask via softmax kernel (we'll put -inf for j<=i before softmax)
        # Softmax over j (S dimension) per (b,h,i)
        S_masked = torch.empty_like(S_attn, dtype=torch.float32)

        grid_softmax = (total_rows,)
        softmax_rows_kernel[grid_softmax](
            S_attn, S_masked,
            B, S, Hq,
            S_attn.stride(0), S_attn.stride(1), S_attn.stride(2), S_attn.stride(3),
            S_masked.stride(0), S_masked.stride(1), S_masked.stride(2), S_masked.stride(3),
        )

        # 7) Output projection (no bias) in Triton: Attn_output = S_masked @ o_proj_weight^T
        D_out = D_q  # Hq * HEAD_DIM
        # S_masked shape [B*S*Hq, HEAD_DIM]; o_proj_weight: [D_out, D_out]
        In = S_masked.view(B * S * Hq, HEAD_DIM)  # [M, K]
        Out = torch.empty((B * S * Hq, D_out), device=device, dtype=torch.float32)

        M = B * S * Hq
        N = D_out
        K = HEAD_DIM

        # Launch output projection kernel
        grid_out = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        linear_output_kernel[grid_out](
            In, o_proj_weight, Out,
            M, N, K,
            In.stride(0), In.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out.stride(0), Out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # Reshape back to [B, S, D_out] with D_out = 12288
        output = Out.view(B, S, D_out)

        # Ensure dtype matches input's expected dtype (float32)
        # The original code returns float32; we return float32 and contiguous
        return output.contiguous()


def run(*args):
    return ModelNew()(*args)
