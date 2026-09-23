import torch
import triton
import triton.language as tl

# Constants
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)

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

    # Add bias if provided
    if Bias_ptr != 0:
        bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BN]
        acc += bias[None, :]  # broadcast over rows

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K:
# Normalize along dim=-1 of a [B, S, H, D] tensor for each (b,s,h), then multiply by weight [H*D]
@triton.jit
def rms_norm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,  # weight stride along H*D (contiguous, stride=1)
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)  # over B*S*H
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # mean over D
    sum_sq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + RMS_EPS)  # 1/sqrt(var + eps)

    # normalize and apply weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms
        w = tl.load(Weight_ptr + h * D + offs_d * stride_w, mask=mask, other=1.0)  # weight is [H*D] contiguous
        y = y * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton rotate half for last dimension (HEAD_DIM split into 2x64): for each (b, s, h), read Q/K and write rotated
@triton.jit
def rotate_half_kernel(
    Input_ptr, Cos_ptr, Sin_ptr, Output_ptr,
    B, S, H, D,
    stride_ib, stride_is, stride_ih, stride_id,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)  # over B*S*H
    total = B * S * H
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    half = D // 2
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D

        first = offs_d < half  # first half indices
        # Load q1 and q2 for both halves
        q1 = tl.load(Input_ptr + b * stride_ib + s * stride_is + h * stride_ih + (offs_d % half) * stride_id, mask=mask & first, other=0.0)
        q2 = tl.load(Input_ptr + b * stride_ib + s * stride_is + h * stride_ih + (offs_d - half), mask=mask & (~first), other=0.0)

        # Combine both halves: q1 in first half, q2 in second half (relative to original D)
        q = tl.zeros((BLOCK_D,), dtype=tl.float32)
        q = tl.where(first, q1, q)
        q = tl.where(~first, q2, q)

        cos = tl.load(Cos_ptr + offs_d, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(Sin_ptr + offs_d, mask=mask, other=1.0).to(tl.float32)

        # Rotate: cat((-q2, q1), dim=-1) -> for q1 positions (d<64): cos*q1 + sin*q2, for q2 positions (d>=64): cos*q2 - sin*q1
        q_rot = tl.zeros((BLOCK_D,), dtype=tl.float32)
        q_rot = tl.where(first, cos * q1 + sin * q2, q_rot)
        q_rot = tl.where(~first, cos * q2 - sin * q1, q_rot)

        tl.store(Output_ptr + b * stride_ob + s * stride_os + h * stride_oh + offs_d * stride_od, q_rot, mask=mask)

# 4) Triton GQA expansion: expand K/V from Hk=8 to Hq=96 by repeating along NUM_KEY_VALUE_GROUPS=12
#   Input: X [B, Hk, S, D], Output: Y [B, Hq, S, D], mapping: h_out -> group_id in [0..11], h_in = h_out % Hk
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, Hk, S, D, Hq, G,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)  # over B*Hq
    total = B * Hq
    if pid >= total:
        return
    b = pid // Hq
    h_out = pid % Hq
    group_id = h_out // Hk  # map 96 -> 12 groups, each group repeats Hk=8
    if group_id >= G:
        return  # safety (shouldn't happen if Hq==G*Hk)
    h_in = h_out % Hk

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x = tl.load(X_ptr + b * stride_xb + h_in * stride_xh + offs_s[:, None] * stride_xs + offs_d[None, :] * stride_xd,
                        mask=mask_s[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            # Store to Y at h_out
            tl.store(Y_ptr + b * stride_yb + h_out * stride_yh + offs_s[:, None] * stride_ys + offs_d[None, :] * stride_yd,
                     x, mask=mask_s[:, None] & mask_d[None, :])

# 5) Triton attention scores accumulation: compute attn[b, h, i, j] for all (b,h,i,j)
#   Inputs: Qn [B, S, Hq, D], Kexp [B, S, Hq, D], Outputs: Attn [B, S, Hq, S], write zeros-initialized first
@triton.jit
def attention_scores_kernel(
    Qn_ptr, Kexp_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ab, stride_as, stride_ah, stride_ad,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (B*S*H, ceil_div(S, BLOCK_S))
    pid_row = tl.program_id(0)  # over B*S*H
    pid_col_block = tl.program_id(1)  # over columns in tiles

    total = B * S * H
    if pid_row >= total:
        return
    b = pid_row // (S * H)
    rem = pid_row % (S * H)
    s_i = rem // H
    h = rem % H

    # j tile
    s0 = pid_col_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_j = s0 < S

    # accumulator over j for this row
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # loop over K-dimension in tiles
    for k0 in range(0, D, BLOCK_D):
        offs_k = k0 + tl.arange(0, BLOCK_D)
        mask_k = offs_k < D

        # load Qn[b, s_i, h, offs_k]
        q = tl.load(Qn_ptr + b * stride_qb + s_i * stride_qs + h * stride_qh + offs_k * stride_qd,
                    mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_D]

        # dot product with Kexp[b, s_j in tile, h, offs_k]
        for j in range(0, BLOCK_S):
            s_j = s0[j]
            if s_j < S:
                kvec = tl.load(Kexp_ptr + b * stride_kb + s_j * stride_ks + h * stride_kh + offs_k * stride_kd,
                               mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_D]
                acc[j] += tl.sum(q * kvec, axis=0)  # scalar add

    # scale
    acc *= SCALING

    # store to attn[b, s_i, h, s0]
    a_ptrs = Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + s0 * stride_ad
    tl.store(a_ptrs, acc, mask=mask_j)

# 6) Triton softmax along last dim (j) per row (b,h,i) with causal mask (j <= i -> -inf)
@triton.jit
def softmax_rows_kernel(
    X_ptr, Y_ptr,
    ROWS, N,  # here ROWS = B*S*H, N = S
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)  # over ROWS
    if pid >= ROWS:
        return
    # Compute (b, s, h) from pid
    S = N
    H = N  # not used here because we only have one h per row in softmax; but we don't need H here as we only have S.
    b = pid // (S)
    rem = pid % (S)
    # With N=S, softmax per row pid:
    # Let's just treat it as a vector of length N
    # We need to know which (b, s, h) row we are processing. In attention, each row corresponds to (b,h,i), where i is the softmax dimension (j).
    # But since we pass ROWS=B*S*H and N=S, we can reconstruct:
    # Each row corresponds to a fixed (b, h) and i is the index along N. For simplicity, we consider softmax over N=S per (b,h) row.
    # However, with N=S, we only have S; but we actually need softmax over S for each (b,h,i). Since i is the last dim, we can take i=N.
    # Given our setup, we assume each row is for fixed (b, h) and i=N. This kernel is for softmax along S dimension.
    # So we compute softmax for vector X[b, :, h] of length S. But we don't have h here, so we re-map pid to (b, i=S, h) by using pid as row id and N as length.
    # For correctness, we treat each row as X[b, i=N, :] and Y accordingly, but we need (b, h). To avoid confusion, we instead launch a wrapper that maps pid to (b, h, i).
    # Here, to keep code simple, we launch softmax_rows_kernel with ROWS=B*S*H and N=S, and inside we compute (b, h, i) from pid. But S is also N here.
    # So we redefine: each row corresponds to (b, h, i), where i runs from 0..S-1. We can't recover h from pid since N=S. Therefore, we launch softmax_rows_kernel with ROWS=B*S*H and N=S, but this mapping is not correct.
    # To fix: we must launch softmax over the attention matrix with explicit (b, h) indexing. Triton doesn't provide easy way to read N=S as S for each row without host-side loop.
    # So we will not use this kernel in forward; instead, we implement softmax in Python after Triton computes attention scores by using torch operations. This avoids Triton softmax pitfalls here.

# 7) Triton output projection without bias: Out[b, i, :] = attn_output[b, i, :] @ o_proj_weight[:, :] (no bias)
@triton.jit
def output_projection_kernel(
    X_ptr, W_ptr, Out_ptr,
    M, N, K,  # X: [M, N], W: [K, N], Out: [M, K]
    stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xn)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + (k0 + offs_k)[None, :] * stride_wn)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Optional: softmax_rows_kernel with correct mapping for attention: softmax over j per (b,h,i)
# We will not invoke it here to avoid complexity. Instead, we use torch.softmax in forward on the attention matrix computed by Triton.

class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = 1e-6):
        super().__init__()
        self.rms_eps = rms_eps

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,
                q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor,
                k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor,
                v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor):
        # Ensure everything is on CUDA and float32
        assert hidden_states.is_cuda and q_proj_weight.is_cuda and q_proj_bias.is_cuda, "All tensors must be on CUDA device."
        device = hidden_states.device
        dtype = torch.float32  # original code uses fp32

        B, S_in, K_in = hidden_states.shape
        # Projections: Q, K, V using Triton GEMM + bias
        # Allocate outputs
        Q = torch.empty((B, S_in, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)
        K = torch.empty((B, S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=torch.float32)
        V = torch.empty((B, S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Launch linear_gemm_bias for Q
        grid_q = (triton.cdiv(S_in, 64), triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, 64))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S_in, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )
        # Launch linear_gemm_bias for K
        grid_k = (triton.cdiv(S_in, 64), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, 64))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S_in, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )
        # Launch linear_gemm_bias for V
        grid_v = (triton.cdiv(S_in, 64), triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, 64))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S_in, K_in,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # Reshape to heads
        Q_heads = Q.view(B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM)    # [B, S, 96, 128]
        K_heads = K.view(B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]
        V_heads = V.view(B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]

        # 2) RMSNorm per head over last dim for Q and K
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms_q = (B * S_in * NUM_ATTENTION_HEADS,)
        rms_norm_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            1,  # stride_w is 1 since q_norm_weight is contiguous 1D
            BLOCK_D=64
        )
        grid_rms_k = (B * S_in * NUM_KEY_VALUE_HEADS,)
        rms_norm_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            1,
            BLOCK_D=64
        )

        # 3) Rotate half for Q and K
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rotate_q = (B * S_in * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rotate_q](
            Q_norm, cos, sin, Q_rot,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=64
        )
        grid_rotate_k = (B * S_in * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_k](
            K_norm, cos, sin, K_rot,
            B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=64
        )

        # 4) GQA expansion: expand K and V from 8 heads to 96 heads (repeat along NUM_KEY_VALUE_GROUPS=12)
        K_expanded = torch.empty((B, NUM_ATTENTION_HEADS, S_in, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, NUM_ATTENTION_HEADS, S_in, HEAD_DIM), device=device, dtype=torch.float32)

        grid_expand = (B * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_expand](
            K_rot, K_expanded,
            B, NUM_KEY_VALUE_HEADS, S_in, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )
        gqa_expand_kernel[grid_expand](
            V_heads, V_expanded,
            B, NUM_KEY_VALUE_HEADS, S_in, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )

        # 5) Compute attention scores using Triton: attn[b, h, i, j] = sum_k Q_rot[b, h, i, k] * K_expanded[b, h, j, k] * SCALING
        Attn = torch.empty((B, S_in, NUM_ATTENTION_HEADS, S_in), device=device, dtype=torch.float32)

        grid_attn = (B * S_in * NUM_ATTENTION_HEADS, triton.cdiv(S_in, 64))
        attention_scores_kernel[grid_attn](
            Q_rot, K_expanded, Attn,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )

        # 6) Apply causal mask and softmax along last dim (j) per (b, h, i) using torch (to avoid Triton softmax pitfalls here)
        causal_mask = torch.triu(torch.full((S_in, S_in), float('-inf'), device=device, dtype=torch.float32), diagonal=1)
        # Expand causal_mask to [B, S, H, S]
        for b in range(B):
            for h in range(NUM_ATTENTION_HEADS):
                # Attn[b, h, :, :] += causal_mask
                Attn[b, :, h, :] = Attn[b, :, h, :] + causal_mask

        attn_softmax = torch.softmax(Attn, dim=-1)  # [B, S, H, S]

        # 7) Compute attention output: attn_output[b, i, :] = sum_j attn_softmax[b, i, j] * V_expanded[b, i, j, :]
        # We need a temporary tensor of shape [B,


def run(*args):
    return ModelNew()(*args)
