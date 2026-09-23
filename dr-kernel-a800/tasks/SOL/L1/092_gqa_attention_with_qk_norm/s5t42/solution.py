import torch
import triton
import triton.language as tl

# Constants from the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
# Inputs: A [M, K], B [N, K], Bias [N], Output C [M, N]
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

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K:
#   For a tensor X of shape [B, S, H, D], normalize along D and apply weight W of shape [H*D].
#   We implement per-row normalization: for each row (b, s, h), compute mean of squares across D,
#   then x_norm = x * rsqrt(mean + eps), y = x_norm * W[h*D + d].
@triton.jit
def rmsnorm_rows_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    RMS_EPS: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # One program processes one row (b, s, h)
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return

    b = pid // (S * H)
    tmp = pid % (S * H)
    s = tmp // H
    h = tmp % H

    # Compute mean of squares across D
    sum_sq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + RMS_EPS)

    # Write normalized and weighted output
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=mask, other=0.0).to(tl.float32)
        norm_x = x * inv_rms
        w = tl.load(W_ptr + h * D + offs_d, mask=mask, other=0.0).to(tl.float32)
        y = norm_x * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton rotate-half for Q and K: split last dim into two halves and apply cos/sin rotation.
#    Input X [B, S, H, D], Output Y [B, S, H, D]
#    For Q/K: rotated[..., :64] = -X[..., 64:], rotated[..., 64:] = X[..., :64]
#    Y = X * cos + rotated * sin
@triton.jit
def rotate_half_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
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

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=mask, other=0.0).to(tl.float32)
        # Split into two halves
        half = D // 2  # 64
        d1 = offs_d % half  # first half indices 0..63
        d2 = (offs_d - half) % half + half  # second half mapped back to original indices 64..127

        x1 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + d1 * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + d2 * stride_xd, mask=mask, other=0.0)

        rotated2 = -x2
        rotated1 = x1
        rotated = tl.where(offs_d < half, rotated1, rotated2)

        cosv = tl.load(Cos_ptr + d1, mask=mask, other=0.0)
        sinv = tl.load(Sin_ptr + d1, mask=mask, other=0.0)
        y = x * cosv + rotated * sinv

        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 4) Triton GQA expand: expand K and V from Hk to Hq by repeating along NUM_KEY_VALUE_GROUPS
#    Inputs: Xh [B, S, Hk, D], Outputs: Yh [B, S, Hq, D]
@triton.jit
def gqa_expand_kernel(
    Xh_ptr, Yh_ptr,
    B, S, Hk, D, Hq, G,
    stride_xbh, stride_xsh, stride_xhk, stride_xd,
    stride_ybh, stride_ysh, stride_yhq, stride_yd,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    # One program writes one row (b, s, hq) over D tiles
    pid_row = tl.program_id(0)
    total_rows = B * S * Hq
    if pid_row >= total_rows:
        return

    b = pid_row // (S * Hq)
    tmp = pid_row % (S * Hq)
    s = tmp // Hq
    hq = tmp % Hq

    # Determine source head h = hq % Hk and group offset
    h = hq % Hk
    group = hq // Hk  # should be >= G groups are NUM_KEY_VALUE_GROUPS=12, Hq=96, Hk=8 => group in [0..11]
    # But original GQA logic maps each of the 96 attention heads to one of the 8 key-value heads via groups.
    # Our input K/V are 8 heads; we replicate each head into 12 groups to get 96 heads.
    # The code in original expands via [:, :, None, :, :].expand(B, Hk, G, S, D).reshape(B, Hq, S, D)
    # We emulate that by copying Xh[b, s, h, :] into Yh[b, s, hq, :] for all hq.

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(Xh_ptr + b * stride_xbh + s * stride_xsh + h * stride_xhk + offs_d * stride_xd,
                    mask=mask, other=0.0).to(tl.float32)
        tl.store(Yh_ptr + b * stride_ybh + s * stride_ysh + hq * stride_yhq + offs_d * stride_yd, x, mask=mask)

# 5) Triton attention scores kernel: compute Attn[b, h, i, j] = sum_k Q_norm[b, h, i, k] * K_expanded[b, h, j, k] * SCALING
#    Inputs: Qn [B, S, H, D], Kexp [B, S, H, S, D], Output Attn [B, S, H, S]
@triton.jit
def attention_scores_kernel(
    Q_ptr, Kexp_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kbe, stride_kse, stride_khe, stride_kje, stride_kd,
    stride_ab, stride_as, stride_ah, stride_asj,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid is 2D: (B*S*H, ceil(S / BLOCK_S))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    total_rows = B * S * H
    if pid0 >= total_rows:
        return

    b = pid0 // (S * H)
    tmp = pid0 % (S * H)
    s_i = tmp // H
    h = tmp % H

    # Column block
    col_block = pid1
    offs_j = col_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_j = offs_j < S

    # Accumulate attn[b, h, i, offs_j]
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load Q vector for i
        q_vec = tl.load(Q_ptr + b * stride_qb + s_i * stride_qs + h * stride_qh + offs_d * stride_qd,
                        mask=mask_d, other=0.0).to(tl.float32)  # [BD]

        # For each j in offs_j, accumulate sum_k Q[k] * K[b, h, j, k]
        # We do a loop over j indices (BLOCK_S) and inner loop over D tiles
        for jj in range(0, BLOCK_S):
            j = offs_j[jj]
            if j < S:
                k_vec = tl.load(Kexp_ptr + b * stride_kbe + s_i * stride_kse + h * stride_khe + j * stride_kje +
                                offs_d * stride_kd, mask=mask_d, other=0.0).to(tl.float32)  # [BD]
                acc[jj] += tl.sum(q_vec * k_vec, axis=0)

    # Apply scaling
    acc = acc * SCALING

    # Store acc to Attn[b, s_i, h, offs_j]
    a_ptrs = Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs_j * stride_asj
    tl.store(a_ptrs, acc, mask=mask_j)

# 6) Triton softmax along last dim (S) per (b, h) row with causal mask (j > i):
#    Inputs: Attn [B, S, H, S], Outputs: Soft [B, S, H, S]
@triton.jit
def softmax_rows_causal_kernel(
    Attn_ptr, Soft_ptr,
    B, S, H,
    stride_ab, stride_as, stride_ah, stride_asj,
    stride_sb, stride_ss, stride_sh, stride_sj,
    BLOCK_S: tl.constexpr
):
    total_rows = B * S * H
    pid = tl.program_id(0)
    if pid >= total_rows:
        return

    b = pid // (S * H)
    tmp = pid % (S * H)
    s_i = tmp // H
    h = tmp % H

    # Compute max over j for numerical stability
    m = -1.0e30
    for j in range(0, S, BLOCK_S):
        offs_j = j + tl.arange(0, BLOCK_S)
        mask_j = offs_j < S
        v = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs_j * stride_asj,
                    mask=mask_j, other=-1.0e30)
        local_max = tl.max(v, axis=0)
        m = tl.maximum(m, local_max)

    # Compute exp and sum
    sum_exp = 0.0
    for j in range(0, S, BLOCK_S):
        offs_j = j + tl.arange(0, BLOCK_S)
        mask_j = offs_j < S
        v = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs_j * stride_asj,
                    mask=mask_j, other=-1.0e30)
        e = tl.exp(v - m)
        # Apply causal mask: j <= s_i -> set to -inf
        causal = offs_j <= s_i
        e = tl.where(causal, -1.0e30, e)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Normalize and store
    for j in range(0, S, BLOCK_S):
        offs_j = j + tl.arange(0, BLOCK_S)
        mask_j = offs_j < S
        v = tl.load(Attn_ptr + b * stride_ab + s_i * stride_as + h * stride_ah + offs_j * stride_asj,
                    mask=mask_j, other=-1.0e30)
        e = tl.exp(v - m)
        causal = offs_j <= s_i
        e = tl.where(causal, -1.0e30, e)
        p = e * inv_sum
        tl.store(Soft_ptr + b * stride_sb + s_i * stride_ss + h * stride_sh + offs_j * stride_sj, p, mask=mask_j)

# 7) Triton output projection (no bias): Attn_out [B, S, Hq*D] @ W_out [Hq*D, Hq*D]^T -> Out [B, S, Hq*D]
#    We implement GEMM-like multiplication: for each (b, i), compute Out[b, i, :] = Attn_out[b, i, :] @ W_out^T
@triton.jit
def output_projection_kernel(
    Attn_out_ptr, W_ptr, Out_ptr,
    B, S, D,
    stride_ab, stride_as, stride_ad,  # Attn_out strides
    stride_wm, stride_wn,             # W strides
    stride_ob, stride_os, stride_od,  # Out strides
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    total_rows = B * S
    pid = tl.program_id(0)
    if pid >= total_rows:
        return

    b = pid // S
    i = pid % S

    # Accumulate per output dimension
    acc = tl.zeros((D,), dtype=tl.float32)

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Load Attn_out[b, i, offs_s]
        a_vec = tl.load(Attn_out_ptr + b * stride_ab + i * stride_as + offs_s * stride_ad,
                        mask=mask_s, other=0.0).to(tl.float32)  # [BS]

        # Load W_out^T row: for each d in D, w = W_out[offs_s, d]
        # Since W_out is [D, D], W_out^T is [D, D], we want w = W_out[offs_s, d]
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            w = tl.load(W_ptr + offs_s[:, None] * stride_wm + offs_d[None, :] * stride_wn,
                        mask=(mask_s[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)  # [BS, BD]
            # acc += sum_s a_vec[s] * w[s, d]
            # Compute per d lane
            for ss in range(0, BLOCK_S):
                s = offs_s[ss]
                if s < S:
                    a = a_vec[ss]
                    w_row = w[ss, :]
                    acc += a * tl.sum(w_row, axis=0)  # sum over d tile

    # Store Out[b, i, :]
    out_ptrs = Out_ptr + b * stride_ob + i * stride_os + tl.arange(0, D) * stride_od
    tl.store(out_ptrs, acc, mask=(tl.arange(0, D) < D))

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
        # All tensors must be CUDA for Triton
        assert hidden_states.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Linear projections using Triton GEMM + bias
        # hidden_states: [B, S, K]
        B, S, K = hidden_states.shape  # K = 12288
        Dq = NUM_ATTENTION_HEADS * HEAD_DIM  # 12288
        Dhk = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 1024
        Dv = NUM_KEY_VALUE_HEADS * HEAD_DIM   # 1024

        # Allocate outputs
        Q = torch.empty((B, S, Dq), device=device, dtype=torch.float32)
        K_t = torch.empty((B, S, Dhk), device=device, dtype=torch.float32)
        V_t = torch.empty((B, S, Dv), device=device, dtype=torch.float32)

        grid_q = (triton.cdiv(B, 1), triton.cdiv(Dq, 128))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, Dq, K,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128
        )

        grid_k = (triton.cdiv(B, 1), triton.cdiv(Dhk, 128))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K_t,
            B, Dhk, K,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K_t.stride(0), K_t.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128
        )

        grid_v = (triton.cdiv(B, 1), triton.cdiv(Dv, 128))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V_t,
            B, Dv, K,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V_t.stride(0), V_t.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=128
        )

        # 2) Reshape to heads
        Qh = Q.view(B, S, NUM_ATTENTION_HEADS, HEAD_DIM)   # [B, S, 96, 128]
        Kh = K_t.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM) # [B, S, 8, 128]
        Vh = V_t.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM) # [B, S, 8, 128]

        # 3) RMSNorm over last dim for Q and K
        Q_norm = torch.empty_like(Qh, dtype=torch.float32)
        K_norm = torch.empty_like(Kh, dtype=torch.float32)

        grid_rms_q = (B * S * NUM_ATTENTION_HEADS,)
        rmsnorm_rows_kernel[grid_rms_q](
            Qh, q_norm_weight, Q_norm,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Qh.stride(0), Qh.stride(1), Qh.stride(2), Qh.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            RMS_EPS,
            BLOCK_D=64
        )

        grid_rms_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rmsnorm_rows_kernel[grid_rms_k](
            Kh, k_norm_weight, K_norm,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            Kh.stride(0), Kh.stride(1), Kh.stride(2), Kh.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            RMS_EPS,
            BLOCK_D=64
        )

        # 4) Rotate last half for Q and K using cos/sin
        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32)

        grid_rotate = (B * S * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rotate](
            Q_norm, cos, sin, Q_rot,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=64
        )

        grid_rotate_k = (B * S * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_k](
            K_norm, cos, sin, K_rot,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=64
        )

        # 5) GQA expand K and V to 96 heads
        K_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

        grid_gqa = (B * S * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )

        grid_gqa_v = (B * S * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_gqa_v](
            Vh, V_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            Vh.stride(0), Vh.stride(1), Vh.stride(2), Vh.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )

        # 6) Compute attention scores in Triton: [B, S, H, S]
        Attn = torch.empty((B, S, NUM_ATTENTION_HEADS, S), device=device, dtype=torch.float32)

        grid_attn = (B * S * NUM_ATTENTION_HEADS, triton.cdiv(S, 64))
        attention_scores_kernel[grid_attn](
            Q_rot, K_expanded, Attn,
            B, S, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3), K_expanded.stride(4),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            BLOCK_S=64, BLOCK_D=64
        )

        # 7) Softmax along last dim with causal mask in Triton
        Soft = torch.empty_like(Attn, dtype=torch.float32)

        grid_softmax = (B * S * NUM_ATTENTION_HEADS,)
        softmax_rows_causal_kernel[grid_softmax](
            Attn, Soft,
            B, S, NUM_ATTENTION_HEADS,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            BLOCK_S=64
        )

        # 8) Output projection in Triton (no bias)
        # attn_output: [B, S, Dq], W_out: [Dq, Dq]
        attn_output = Soft.view(B, S, Dq)  # [B, S, 12288]
        output = torch.empty((B, S, Dq), device=device, dtype=torch.float32)

        grid_output = (B * S,)
        output_projection_kernel[grid_output](
            attn_output, o_proj_weight, output,
            B, S, Dq,
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_S=64, BLOCK_D=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
