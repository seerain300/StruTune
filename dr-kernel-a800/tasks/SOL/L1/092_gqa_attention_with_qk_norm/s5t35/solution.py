import torch
import triton
import triton.language as tl

# Constants matching the original code setup
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
# A: [M, K], B: [N, K], Bias: [N], C: [M, N]
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
# Input X: [B, S, H, D] (we pass the head as a linear index over BS*H), Weight: [H], Output: Y
# We normalize along D=128 for each (b,s,h).
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,  # Weight is 1D [H]
    BLOCK_D: tl.constexpr
):
    total_rows = B * S * H
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    offs_d = tl.arange(0, BLOCK_D)

    # Load X[b,s,h,:] vector of length D
    x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                mask=(offs_d < D), other=0.0).to(tl.float32)

    # Compute mean of squares over D
    sq = x * x
    mean = tl.sum(sq, axis=0) / D
    inv_rms = tl.rsqrt(mean + RMS_EPS)

    # Normalize
    y = x * inv_rms  # [BLOCK_D]
    gamma = tl.load(Weight_ptr + h * stride_w)  # scalar
    y = y * gamma

    # Store
    y_ptrs = Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd
    tl.store(y_ptrs, y, mask=(offs_d < D))

# 3) Triton rotation for half-dimension: rotate_half(x): first half unchanged, second half negated and swapped.
# We implement rotate for Q and K, each of shape [B, S, H, D], using grid over rows.
@triton.jit
def rotate_half_kernel(
    X_ptr, Out_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_D: tl.constexpr
):
    total_rows = B * S * H
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    offs_d = tl.arange(0, BLOCK_D)

    # Load full vector
    x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                mask=(offs_d < D), other=0.0).to(tl.float32)

    # Split into halves
    first = x[:64]
    second = x[64:]

    # Swap and negate second half
    rotated = tl.concatenate([first, -second], axis=0)

    # Store
    out_ptrs = Out_ptr + b * stride_ob + s * stride_os + h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, rotated, mask=(offs_d < D))

# 4) Triton GQA expansion: K_expanded/Broadcast: expand [B, S, KVH, D] -> [B, S, NH, D] where NH=NUM_ATTENTION_HEADS=96
# We replicate along groups NUM_KEY_VALUE_GROUPS=12, i.e., repeat each KV head 12 times across NH groups.
@triton.jit
def gqa_expand_kernel(
    K_ptr, Out_ptr,
    B, S, KVH, D,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ob, stride_os, stride_oh, stride_od,
    GROUPS: tl.constexpr
):
    total_rows = B * S * KVH * GROUPS
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    # Map pid to (b, s, kvh, group)
    b = pid // (S * KVH * GROUPS)
    rem = pid % (S * KVH * GROUPS)
    s = rem // (KVH * GROUPS)
    kvh = rem // GROUPS % KVH
    g = rem % GROUPS

    nh = kvh * GROUPS + g  # nh in [0..96)

    # Compute K index for kvh and copy to nh
    k_ptrs = K_ptr + b * stride_kb + s * stride_ks + kvh * stride_kh + tl.arange(0, D) * stride_kd
    k_vals = tl.load(k_ptrs, mask=(tl.arange(0, D) < D), other=0.0).to(tl.float32)

    out_ptrs = Out_ptr + b * stride_ob + s * stride_os + nh * stride_oh + tl.arange(0, D) * stride_od
    tl.store(out_ptrs, k_vals, mask=(tl.arange(0, D) < D))

# 5) Triton attention score kernel: compute attn[b, h, i, j] = sum_k Q_norm[b, h, i, k] * K_expanded[b, h, j, k] * SCALING
# Inputs: Q_norm [B, S, H, D], K_expanded [B, S, H, D], Outputs: Attn [B, S, H, S] initialized to -inf
# We accumulate and store for each (b,i,h). j varies implicitly via grid; we choose grid (B*S*H,).
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ab, stride_as, stride_ah, stride_ad,
    BLOCK_D: tl.constexpr
):
    total_rows = B * S * H
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s_i = rem // H  # i index in [0..S)
    h = rem % H

    # Initialize attn row [S] to -inf
    offs_j = tl.arange(0, BLOCK_D)  # we use BLOCK_D=S, but mask ensures validity
    attn_row = tl.full((BLOCK_D,), -float('inf'), dtype=tl.float32)

    # Accumulate over k dimension
    for k0 in range(0, D, BLOCK_D):
        offs_k = k0 + tl.arange(0, BLOCK_D)
        mask_k = offs_k < D

        # Load Q[b, s_i, h, k]
        q_ptrs = Q_ptr + b * stride_qb + s_i * stride_qs + h * stride_qh + offs_k * stride_qd
        q_vec = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load K[b, s_j, h, k] for all s_j in [0..S-1]
        # We need a 2D accumulation: for each j in tile, sum q_vec * K[b, j, h, offs_k]
        # Build K pointers as (S, BLOCK_D)
        attn_partial = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for j in range(0, BLOCK_D):
            # jth index along S; we assume BLOCK_D == S for efficiency; mask ensures bounds.
            j_idx = j  # only valid when BLOCK_D == S; we guard by comparing j_idx < S
            if j_idx < S:
                k_ptrs = K_ptr + b * stride_kb + j_idx * stride_ks + h * stride_kh + offs_k * stride_kd
                k_vec = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
                attn_partial += q_vec * k_vec

        attn_row += attn_partial

    # Scale by SCALING and store into Attn[b, s_i, h, :]
    attn_out_ptrs = Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs_j * stride_ad
    tl.store(attn_out_ptrs, attn_row * SCALING, mask=(offs_j < S))

# 6) Triton softmax per row along last dim (S) for causal mask: attn[b, h, i, j] = -inf if j <= i
# We apply softmax over j per (b, h, i) and write to Soft[b, s_i, h, :]
@triton.jit
def softmax_rows_kernel(
    Attn_ptr, Soft_ptr,
    B, S, H,
    stride_ab, stride_as, stride_ah, stride_ad,
    stride_sb, stride_ss, stride_sh, stride_sd,
    BLOCK_S: tl.constexpr
):
    total_rows = B * S * H
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s_i = rem // H  # i index
    h = rem % H

    offs = tl.arange(0, BLOCK_S)  # we assume BLOCK_S == S
    # Load row attn[b, h, s_i, :]
    attn_row_ptrs = Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs * stride_ad
    attn_row = tl.load(attn_row_ptrs, mask=(offs < S), other=-float('inf')).to(tl.float32)

    # Apply causal mask: j <= i -> attn_row[j] = -inf
    # Triton allows scalar condition broadcasting
    causal_mask = (offs <= s_i)
    attn_row = tl.where(causal_mask, -float('inf'), attn_row)

    # Max for numerical stability
    max_val = tl.max(attn_row, axis=0)
    # exp
    exp_row = tl.exp(attn_row - max_val)
    # sum
    sum_val = tl.sum(exp_row, axis=0)
    # normalize
    soft_row = exp_row / sum_val

    # Store to Soft[b, s_i, h, :]
    soft_row_ptrs = Soft_ptr + b * stride_sb + s_i * stride_ss + h * stride_sh + offs * stride_sd
    tl.store(soft_row_ptrs, soft_row, mask=(offs < S))

# 7) Triton output projection: Out[M, P] = Attn[M, K] @ W[P, K]^T (no bias), where M=B*S*H, K=Dq=12288, P=Dq.
# Implement GEMM without bias.
@triton.jit
def output_projection_kernel(
    Attn_ptr, W_ptr, Out_ptr,
    M, K, P,
    stride_am, stride_ak,
    stride_wk, stride_wp,
    stride_om, stride_op,
    BLOCK_M: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = Attn_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)  # [BM, BK]
        w_ptrs = W_ptr + (offs_k[:, None] + (k + offs_k)[None, :]) * stride_wk + offs_p[None, :] * stride_wp  # [BK, BP]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_p[None, :] < P), other=0.0)

        acc += tl.dot(a, w)  # [BM, BP]

    # Store
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_p[None, :] * stride_op)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_p[None, :] < P))

# Entry point: ModelNew
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
        # Ensure CUDA tensors
        assert hidden_states.is_cuda, "All tensors must be on CUDA for Triton."
        device = hidden_states.device
        dtype = torch.float32  # original run uses fp32

        B, S, K_in = hidden_states.shape  # K_in = 3 * 12288 if original, but here hidden_states is [B, S, 12288] as per code
        # Create linear projections using Triton
        # Q: [B, S, NUM_ATTENTION_HEADS*HEAD_DIM] = [B, S, 12288]
        Q = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        K = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)
        V = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)

        # Grid for linear_gemm_bias_kernel
        Mq = B * S
        Nq = NUM_ATTENTION_HEADS * HEAD_DIM
        Kq = hidden_states.shape[2]  # 12288
        grid_q = (triton.cdiv(Mq, 128), triton.cdiv(Nq, 128))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            Mq, Nq, Kq,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        # Same for K and V
        Mkv = B * S
        Nkv = NUM_KEY_VALUE_HEADS * HEAD_DIM
        Kkv = hidden_states.shape[2]
        grid_kv = (triton.cdiv(Mkv, 128), triton.cdiv(Nkv, 128))
        linear_gemm_bias_kernel[grid_kv](
            hidden_states, k_proj_weight, k_proj_bias, K,
            Mkv, Nkv, Kkv,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        linear_gemm_bias_kernel[grid_kv](
            hidden_states, v_proj_weight, v_proj_bias, V,
            Mkv, Nkv, Kkv,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, NUM_ATTENTION_HEADS, HEAD_DIM)    # [B, S, 96, 128]
        K_heads = K.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]
        V_heads = V.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]

        # 3) RMSNorm per head (over last dim=128) for Q and K
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms_q = (B * S * NUM_ATTENTION_HEADS,)
        rmsnorm_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0),  # weight is 1D [H]
            BLOCK_D=128
        )

        grid_rms_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rmsnorm_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0),
            BLOCK_D=128
        )

        # 4) Rotate half for Q and K (swap first 64 with negated second half)
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rotate_q = (B * S * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rotate_q](
            Q_norm, Q_rot,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=128
        )

        grid_rotate_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_k](
            K_norm, K_rot,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=128
        )

        # 5) GQA: expand K and V from 8 heads to 96 heads by repeating groups
        K_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

        grid_gqa = (B * S * NUM_KEY_VALUE_HEADS * NUM_KEY_VALUE_GROUPS,)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            GROUPS=NUM_KEY_VALUE_GROUPS
        )

        gqa_expand_kernel[grid_gqa](
            V_heads, V_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            GROUPS=NUM_KEY_VALUE_GROUPS
        )

        # 6) Compute attention scores: attn[b, h, i, j] = sum_k Q_rot[b,h,i,k] * K_expanded[b,h,j,k] * SCALING
        Attn = torch.empty((B, S, NUM_ATTENTION_HEADS, S), device=device, dtype=torch.float32)

        grid_attn = (B * S * NUM_ATTENTION_HEADS,)
        attn_scores_kernel[grid_attn](
            Q_rot, K_expanded, Attn,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            BLOCK_D=S  # S is 128 in provided configs
        )

        # 7) Softmax over j per row (b, h, i) with causal mask (j <= i -> -inf)
        Soft = torch.empty_like(Attn, dtype=torch.float32)

        grid_softmax = (B * S * NUM_ATTENTION_HEADS,)
        softmax_rows_kernel[grid_softmax](
            Attn, Soft,
            B, S, NUM_ATTENTION_HEADS,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            BLOCK_S=S
        )

        # 8) Compute attention output: attn_output[b, i, :] = Soft[b,i,:] @ V_expanded[b,:,i,:]
        # We need to produce [B, S, Dq] where Dq = NUM_ATTENTION_HEADS * HEAD_DIM = 12288
        attn_output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Implement GEMM in Triton: Out[M, P] = Soft[M, K] @ V_exp[P, K]^T, M=B*S, K=S, P=Dq
        M = B * S
        K = S
        P = NUM_ATTENTION_HEADS * HEAD_DIM
        grid_out = (triton.cdiv(M, 128), triton.cdiv(P, 128))
        output_projection_kernel[grid_out](
            Soft, V_expanded,
            attn_output,
            M, K, P,
            Soft.stride(0), Soft.stride(2),  # Soft[b, s_i, h, :] -> strides over rows and K (last dim)
            V_expanded.stride(2), V_expanded.stride(3),  # over K and P
            attn_output.stride(0), attn_output.stride(1),
            BLOCK_M=128, BLOCK_P=128, BLOCK_K=64
        )

        # 9) Output projection (no bias): match original F.linear(..., None)
        # attn_output [B, S, 12288] -> output [B, S, 12288]
        output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)
        M_out = B * S
        P_out = NUM_ATTENTION_HEADS * HEAD_DIM
        grid_last = (triton.cdiv(M_out, 128), triton.cdiv(P_out, 128))
        output_projection_kernel[grid_last](
            attn_output, o_proj_weight,
            output,
            M_out, P_out, P_out,  # K=P_out
            attn_output.stride(0), attn_output.stride(1),
            o_proj_weight.stride(1), o_proj_weight.stride(0),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_P=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
