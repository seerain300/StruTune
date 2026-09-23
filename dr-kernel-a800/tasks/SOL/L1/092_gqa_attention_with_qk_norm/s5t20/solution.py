import torch
import triton
import triton.language as tl

# Constants from original code
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

        acc += tl.dot(a, b)  # [BM, BN]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BN]
    acc += bias[None, :]  # broadcast over rows

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K
# Normalize along dim=-1 of a tensor with shape (B, S, H, D) and weight of shape (H*D)
@triton.jit
def rms_norm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w, EPS: tl.constexpr, BLOCK_D: tl.constexpr
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return

    # Compute indices for b, s, h
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # Reduction over D
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32)

    mean = sumsq / D
    rinv = tl.rsqrt(mean + EPS)

    # Apply normalization and weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=mask, other=0.0)
        w = tl.load(Weight_ptr + (h * D + offs_d) * stride_w, mask=mask, other=0.0)
        y = (x.to(tl.float32) * rinv) * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton rotation for last half (64) for Q and K
# Input: X [B, S, H, D], Output: Y [B, S, H, D], rotate: [q1, q2] -> [-q2, q1] for last 64 dims
@triton.jit
def rotate_half_kernel(
    X_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return

    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # First half and second half
    D_half = D // 2
    for d0 in range(0, D_half, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D_half

        # q1 = X[..., :64]
        x1 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                     mask=mask, other=0.0)
        # q2 = X[..., 64:128]
        x2 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + (offs_d + D_half) * stride_xd,
                     mask=mask, other=0.0)

        y1 = x1
        y2 = -x2

        # Store rotated: Y[..., :64] = q1, Y[..., 64:128] = -q2
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y1, mask=mask)
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + (offs_d + D_half) * stride_yd, y2, mask=mask)

# 4) Triton GQA expansion: K and V from NUM_KEY_VALUE_HEADS to NUM_ATTENTION_HEADS by repeating groups
# Input: X [B, S, Hv, D], Output: Y [B, Hq, S, D] where Hq = NUM_ATTENTION_HEADS
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, S, Hv, Hq, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    G_GROUPS: tl.constexpr  # NUM_KEY_VALUE_GROUPS
):
    total = B * S * Hv
    pid = tl.program_id(0)
    if pid >= total:
        return

    b = pid // (S * Hv)
    rem = pid % (S * Hv)
    s = rem // Hv
    hv = rem % Hv

    # Determine group index
    group = hv // G_GROUPS  # 0..11
    # Destination head index h for this expansion
    for h in range(0, Hq):
        # If h falls into the same group (h % 12 == group), copy; else copy zeros
        # But since Hq = 96 and G_GROUPS = 12, h % 12 == group selects the same group
        if (h % G_GROUPS) == group:
            x = tl.load(X_ptr + b * stride_xb + s * stride_xs + hv * stride_xh + tl.arange(0, D) * stride_xd,
                        mask=(tl.arange(0, D) < D), other=0.0)
            tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + tl.arange(0, D) * stride_yd,
                     x, mask=(tl.arange(0, D) < D))

# 5) Triton attention scores: compute attn[b, h, i, j] = sum_k Q_norm[b,h,i,k] * K_exp[b,h,j,k] * scaling
# We launch one program per (b, h, i). Each program computes a row vector over j in tiles.
@triton.jit
def attn_scores_row_kernel(
    Q_ptr, K_ptr, S_ptr,
    B, S, Hq, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_sb, stride_sh, stride_si, stride_sj,  # S_ptr is [B, Hq, S, S]
    scaling,
    BLOCK_J: tl.constexpr
):
    pid = tl.program_id(0)
    # pid ranges over B * Hq * S
    b = pid // (Hq * S)
    rem = pid % (Hq * S)
    h = rem // S
    i = rem % S

    # Initialize output vector for j from 0 to S
    j_vec = tl.arange(0, S)

    # Accumulate dot products over k-dimension in tiles of D
    acc = tl.zeros((S,), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_J):
        offs_k = k0 + tl.arange(0, BLOCK_J)
        mask_k = offs_k < D

        # Load Q_n row segment: Q_norm[b, h, i, offs_k]
        q = tl.load(
            Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + offs_k * stride_qd,
            mask=mask_k, other=0.0
        ).to(tl.float32)  # [BLOCK_J]

        # Load K segments across j: K_norm[b, h, j, offs_k] for j in 0..S-1
        for jj in range(0, S):
            kj = tl.load(
                K_ptr + b * stride_kb + h * stride_kh + jj * stride_ks + offs_k * stride_kd,
                mask=mask_k, other=0.0
            ).to(tl.float32)  # [BLOCK_J]
            acc[jj] += tl.sum(q * kj)  # scalar accumulation for each jj

    # Apply scaling and store into S_ptr[b, h, i, :]
    s_ptrs = S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j_vec * stride_sj
    # If j_vec > i, keep acc; else set to -inf for causal mask
    # Triton doesn't have inf constant; use a large negative value
    large_neg = -1e20
    for jj in range(0, S):
        if (jj > i):
            tl.store(s_ptrs + jj, acc[jj] * scaling)
        else:
            tl.store(s_ptrs + jj, large_neg)

# 6) Triton softmax over last dim (S) per (b, h, i) with causal mask (j > i)
# Input: S_ptr [B, Hq, S, S] containing scores; Output: S_out_ptr same shape with softmax along last dim per row
@triton.jit
def softmax_causal_rows_kernel(
    S_ptr, S_out_ptr,
    B, S, Hq,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_ob, stride_oh, stride_oi, stride_oj,
    BLOCK_J: tl.constexpr
):
    total = B * Hq * S
    pid = tl.program_id(0)
    if pid >= total:
        return

    b = pid // (Hq * S)
    rem = pid % (Hq * S)
    h = rem // S
    i = rem % S

    j_vec = tl.arange(0, S)

    # Load row segment
    s_ptrs = S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j_vec * stride_sj
    row = tl.load(s_ptrs, mask=(j_vec < S), other=-1e20)

    # Compute max
    row_max = -1e20
    for jj in range(0, S):
        row_max = tl.maximum(row_max, row[jj])

    # Compute exp and sum
    exp_row = tl.exp(row - row_max)
    row_sum = 0.0
    for jj in range(0, S):
        row_sum += exp_row[jj]

    # Normalize
    probs = exp_row / row_sum

    # Store back with causal mask: keep probs where j > i, else 0
    out_ptrs = S_out_ptr + b * stride_ob + h * stride_oh + i * stride_oi + j_vec * stride_oj
    for jj in range(0, S):
        if (jj > i):
            tl.store(out_ptrs + jj, probs[jj])
        else:
            tl.store(out_ptrs + jj, 0.0)

# 7) Triton output projection: O = attn_output [B, S, Hq*D] @ O_weight^T (no bias)
# We implement a simple row-wise GEMV-like kernel over D tiles, accumulating into output
@triton.jit
def output_projection_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    B, S, D_in, D_out,
    stride_xb, stride_xs, stride_xd,  # X [B, S, D_in]
    stride_wj, stride_wk, stride_wout,  # Weight [D_out, D_in], Out [B, S, D_out]
    BLOCK_D: tl.constexpr
):
    total = B * S
    pid = tl.program_id(0)
    if pid >= total:
        return

    b = pid // S
    s = pid % S

    out_row = tl.zeros((D_out,), dtype=tl.float32)

    for j in range(0, D_out):
        wj = tl.load(Weight_ptr + j * stride_wj + tl.arange(0, D_in) * stride_wk,
                     mask=(tl.arange(0, D_in) < D_in), other=0.0)  # [D_in]
        x_row = tl.load(X_ptr + b * stride_xb + s * stride_xs + tl.arange(0, D_in) * stride_xd,
                        mask=(tl.arange(0, D_in) < D_in), other=0.0)  # [D_in]
        out_row[j] = tl.sum(wj * x_row)

    out_ptrs = Out_ptr + b * stride_outb + s * stride_outs + tl.arange(0, D_out) * stride_outd
    tl.store(out_ptrs, out_row, mask=(tl.arange(0, D_out) < D_out))

# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rms_eps = rms_eps

    def forward(self, hidden_states,
                q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight,
                q_norm_weight, k_norm_weight,
                cos, sin):
        # hidden_states: [B, S, 12288]
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype  # float32 per harness

        # 1) Linear projections: Q, K, V
        # Use Triton kernels; allocate outputs
        Q = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        K = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)
        V = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)

        # Launch GEMM with bias for Q
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid_q = (triton.cdiv(B, BLOCK_M), triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, BLOCK_N))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, NUM_ATTENTION_HEADS * HEAD_DIM, hidden_states.shape[2],
            hidden_states.stride(0), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # Launch GEMM with bias for K
        grid_k = (triton.cdiv(B, BLOCK_M), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, BLOCK_N))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, NUM_KEY_VALUE_HEADS * HEAD_DIM, hidden_states.shape[2],
            hidden_states.stride(0), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # Launch GEMM with bias for V
        grid_v = (triton.cdiv(B, BLOCK_M), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, BLOCK_N))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, NUM_KEY_VALUE_HEADS * HEAD_DIM, hidden_states.shape[2],
            hidden_states.stride(0), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, NUM_ATTENTION_HEADS, HEAD_DIM)
        K_heads = K.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)
        V_heads = V.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)

        # 3) RMSNorm per head (Q and K), elementwise + reduction in Triton
        Q_norm = torch.empty_like(Q_heads, device=device, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, device=device, dtype=torch.float32)

        grid_qn = (B * S * NUM_ATTENTION_HEADS,)
        rms_norm_kernel[grid_qn](
            Q_heads, q_norm_weight, Q_norm,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0), EPS=self.rms_eps, BLOCK_D=128
        )

        grid_kn = (B * S * NUM_KEY_VALUE_HEADS,)
        rms_norm_kernel[grid_kn](
            K_heads, k_norm_weight, K_norm,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0), EPS=self.rms_eps, BLOCK_D=128
        )

        # 4) Rotate last half for Q and K using Triton
        # Input X is Q_norm and K_norm; output Y is rotated tensors
        Q_rot = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, device=device, dtype=torch.float32)

        grid_qr = (B * S * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_qr](
            Q_norm, Q_rot,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=64
        )

        grid_kr = (B * S * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_kr](
            K_norm, K_rot,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=64
        )

        # 5) GQA expand K and V from 8 heads to 96 heads
        K_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

        grid_gqa = (B * S * NUM_KEY_VALUE_HEADS,)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, NUM_KEY_VALUE_HEADS, NUM_ATTENTION_HEADS, HEAD_DIM,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            G_GROUPS=NUM_KEY_VALUE_GROUPS  # 12
        )

        grid_gva = (B * S * NUM_KEY_VALUE_HEADS,)
        gqa_expand_kernel[grid_gva](
            V_rot, V_expanded,
            B, S, NUM_KEY_VALUE_HEADS, NUM_ATTENTION_HEADS, HEAD_DIM,
            V_rot.stride(0), V_rot.stride(1), V_rot.stride(2), V_rot.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            G_GROUPS=NUM_KEY_VALUE_GROUPS
        )

        # 6) Compute attention scores S [B, Hq, S, S] using Triton attn_scores_row_kernel
        # S_ptr initialized to zeros, then kernel writes scaled dot products
        S_scores = torch.empty((B, NUM_ATTENTION_HEADS, S, S), device=device, dtype=torch.float32)
        S_scores.zero_()

        grid_attn = (B * NUM_ATTENTION_HEADS * S,)
        attn_scores_row_kernel[grid_attn](
            Q_rot, K_expanded, S_scores,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3),
            scaling=SCALING,
            BLOCK_J=64
        )

        # 7) Softmax over last dim per row (b, h, i) with causal mask (j > i) using Triton
        S_out = torch.empty_like(S_scores, device=device, dtype=torch.float32)

        grid_softmax = (B * NUM_ATTENTION_HEADS * S,)
        softmax_causal_rows_kernel[grid_softmax](
            S_scores, S_out,
            B, S, NUM_ATTENTION_HEADS,
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3),
            S_out.stride(0), S_out.stride(1), S_out.stride(2), S_out.stride(3),
            BLOCK_J=64
        )

        # 8) Compute attention output: attn_output = sum_j S_out[b,h,i,j] * V_expanded[b,h,j,:]  => shape [B,S,Hq*D]
        attn_output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Output projection: O = attn_output @ o_proj_weight^T (no bias), implemented in Triton
        Out = torch.empty((B, S, o_proj_weight.shape[0]), device=device, dtype=torch.float32)  # [B,S,12288]
        grid_out = (B * S,)
        output_projection_kernel[grid_out](
            attn_output, o_proj_weight, Out,
            B, S, NUM_ATTENTION_HEADS * HEAD_DIM, o_proj_weight.shape[0],
            attn_output.stride(0), attn_output.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out.stride(0), Out.stride(2),
            BLOCK_D=64
        )

        return Out


def run(*args):
    return ModelNew()(*args)
