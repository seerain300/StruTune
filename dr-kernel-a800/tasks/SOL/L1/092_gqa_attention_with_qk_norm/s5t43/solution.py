import torch
import triton
import triton.language as tl

# Constants from the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
# Here, A is [B*S, K] with stride_am = K, stride_ak = 1 for row-major.
# B is [N, K] (N=M for Q,K,V), Bias is [N].
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

    # Store result to C[M, N]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton RMSNorm for Q: normalize over last dim (HEAD_DIM) per head, then multiply by q_norm_weight
# Input Qh: [B, S, H, D], weight: [H*D], Output Q_norm: [B, S, H, D]
@triton.jit
def rmsnorm_q_kernel(
    Qh_ptr, Weight_ptr, Qnorm_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_w,
    stride_nqb, stride_nqs, stride_nqh, stride_nqd,
    RMS_EPS: tl.constexpr
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    hs = pid % (S * H)
    h = hs // S
    i = hs % S

    base_q = Qh_ptr + b * stride_qb + i * stride_qs + h * stride_qh
    # Compute mean of squares over D
    sum_sq = 0.0
    for d in range(0, D):
        x = tl.load(base_q + d * stride_qd).to(tl.float32)
        sum_sq += x * x
    mean = sum_sq / D
    r = tl.sqrt(mean + RMS_EPS)
    inv_r = 1.0 / r

    weight_vec = tl.load(Weight_ptr + h * D * stride_w)  # [D] vector

    base_out = Qnorm_ptr + b * stride_nqb + i * stride_nqs + h * stride_nqh
    for d in range(0, D):
        x = tl.load(base_q + d * stride_qd).to(tl.float32)
        y = x * inv_r
        w = tl.load(Weight_ptr + (h * D + d) * stride_w)  # scalar per d
        z = y * w
        tl.store(base_out + d * stride_nqd, z)

# 3) Same RMSNorm for K: input Kh: [B, S, Hk, D], weight: [Hk*D], output K_norm: [B, S, Hk, D]
@triton.jit
def rmsnorm_k_kernel(
    Kh_ptr, Weight_ptr, Knorm_ptr,
    B, S, Hk, D,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_w,
    stride_nb, stride_ns, stride_nh, stride_nd,
    RMS_EPS: tl.constexpr
):
    total = B * S * Hk
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * Hk)
    hs = pid % (S * Hk)
    h = hs // S
    i = hs % S

    base_k = Kh_ptr + b * stride_kb + i * stride_ks + h * stride_kh
    sum_sq = 0.0
    for d in range(0, D):
        x = tl.load(base_k + d * stride_kd).to(tl.float32)
        sum_sq += x * x
    mean = sum_sq / D
    r = tl.sqrt(mean + RMS_EPS)
    inv_r = 1.0 / r

    weight_vec = tl.load(Weight_ptr + h * D * stride_w)  # [D]

    base_out = Knorm_ptr + b * stride_nb + i * stride_ns + h * stride_nh
    for d in range(0, D):
        x = tl.load(base_k + d * stride_kd).to(tl.float32)
        y = x * inv_r
        w = tl.load(Weight_ptr + (h * D + d) * stride_w)
        z = y * w
        tl.store(base_out + d * stride_nd, z)

# 4) Rotate last half of head_dim for Q: Qh_in [B,S,H,128] -> Q_rot [B,S,H,128]
# We compute rotated = cat((-q2, q1), dim=-1) where q1=Qh_in[..., :64], q2=Qh_in[..., 64:],
# and then Q_rot = Qh_in * cos + rotated * sin.
@triton.jit
def rotate_q_half_kernel(
    Qh_ptr, Cos_ptr, Sin_ptr, Qrot_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_c, stride_s,
    stride_rb, stride_rs, stride_rh, stride_rd,
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    hs = pid % (S * H)
    h = hs // S
    i = hs % S

    base_q = Qh_ptr + b * stride_qb + i * stride_qs + h * stride_qh
    c = tl.load(Cos_ptr + 0 * stride_c)
    s = tl.load(Sin_ptr + 0 * stride_s)

    # First half: q1
    for d in range(0, D // 2):
        x = tl.load(base_q + d * stride_qd).to(tl.float32)
        y = tl.load(base_q + (d + D // 2) * stride_qd).to(tl.float32)  # q2
        rotated_d = -y * c + x * s  # rotated[..., :64] element at d
        tl.store(Qrot_ptr + b * stride_rb + i * stride_rs + h * stride_rh + d * stride_rd, rotated_d)

    # Second half: q2
    for d in range(0, D // 2):
        x = tl.load(base_q + d * stride_qd).to(tl.float32)
        y = tl.load(base_q + (d + D // 2) * stride_qd).to(tl.float32)  # q1
        rotated_d = -y * c + x * s  # rotated[..., 64:] element at d
        tl.store(Qrot_ptr + b * stride_rb + i * stride_rs + h * stride_rh + (d + D // 2) * stride_rd, rotated_d)

# 5) Rotate last half for K similarly
@triton.jit
def rotate_k_half_kernel(
    Kh_ptr, Cos_ptr, Sin_ptr, Kr_ptr,
    B, S, Hk, D,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_c, stride_s,
    stride_rb, stride_rs, stride_rh, stride_rd,
):
    total = B * S * Hk
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * Hk)
    hs = pid % (S * Hk)
    h = hs // S
    i = hs % S

    base_k = Kh_ptr + b * stride_kb + i * stride_ks + h * stride_kh
    c = tl.load(Cos_ptr + 0 * stride_c)
    s = tl.load(Sin_ptr + 0 * stride_s)

    # First half: q1
    for d in range(0, D // 2):
        x = tl.load(base_k + d * stride_kd).to(tl.float32)
        y = tl.load(base_k + (d + D // 2) * stride_kd).to(tl.float32)  # q2
        rotated_d = -y * c + x * s
        tl.store(Kr_ptr + b * stride_rb + i * stride_rs + h * stride_rh + d * stride_rd, rotated_d)

    # Second half: q2
    for d in range(0, D // 2):
        x = tl.load(base_k + d * stride_kd).to(tl.float32)
        y = tl.load(base_k + (d + D // 2) * stride_kd).to(tl.float32)  # q1
        rotated_d = -y * c + x * s
        tl.store(Kr_ptr + b * stride_rb + i * stride_rs + h * stride_rh + (d + D // 2) * stride_rd, rotated_d)

# 6) Expand K from Hk heads to Hq heads by repeating along NUM_KEY_VALUE_GROUPS=12
# Input Kh: [B, S, Hk, D], Output Kexp: [B, Hq, S, D], mapping h_out -> h_in = h_out // 12
@triton.jit
def expand_k_kernel(
    Kh_ptr, Kexp_ptr,
    B, S, Hq, Hk, D,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_bexp, stride_sexp, stride_hexp, stride_dexp,
    GROUPS: tl.constexpr
):
    total = B * Hq * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (Hq * S)
    hs = pid % (Hq * S)
    h_out = hs // S
    i = hs % S
    h_in = h_out // GROUPS  # 96 -> 8, one-to-one expansion by groups
    if h_in >= Hk:
        return

    base_k = Kh_ptr + b * stride_kb + i * stride_ks + h_in * stride_kh
    base_exp = Kexp_ptr + b * stride_bexp + i * stride_sexp + h_out * stride_hexp

    for d in range(0, D):
        val = tl.load(base_k + d * stride_kd).to(tl.float32)
        tl.store(base_exp + d * stride_dexp, val)

# Same for V
@triton.jit
def expand_v_kernel(
    Vh_ptr, Vexp_ptr,
    B, S, Hq, Hv, D,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_bexp, stride_sexp, stride_hexp, stride_dexp,
    GROUPS: tl.constexpr
):
    total = B * Hq * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (Hq * S)
    hs = pid % (Hq * S)
    h_out = hs // S
    i = hs % S
    h_in = h_out // GROUPS
    if h_in >= Hv:
        return

    base_v = Vh_ptr + b * stride_vb + i * stride_vs + h_in * stride_vh
    base_exp = Vexp_ptr + b * stride_bexp + i * stride_sexp + h_out * stride_hexp

    for d in range(0, D):
        val = tl.load(base_v + d * stride_vd).to(tl.float32)
        tl.store(base_exp + d * stride_dexp, val)

# 7) Compute attention scores: Attn[b, h, i, j] = sum_k Qnorm_rot[b, h, i, k] * Kexp[b, h, j, k] * SCALING
# Inputs: Qnorm_rot [B, H, S, D], Kexp [B, H, S, D]
# Output: Attn [B, H, S, S]
@triton.jit
def attention_scores_kernel(
    Qn_ptr, Kexp_ptr, Attn_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qi, stride_qd,
    stride_kb, stride_kh, stride_kj, stride_kd,
    stride_attnb, stride_attnh, stride_attni, stride_attnj,
    SCALING: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    # grid: (B*H, ceil(S/BLOCK_S))
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    i = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # vector of j positions for this tile

    # Initialize accumulator [BLOCK_S]
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over k dimension in tiles
    for k0 in range(0, D, BLOCK_D):
        d = k0 + tl.arange(0, BLOCK_D)
        # Load Q vector at (b, h, i, d): shape [BLOCK_D]
        base_q = Qn_ptr + b * stride_qb + h * stride_qh + i * stride_qi  # i is vector [BLOCK_S]
        q_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, BLOCK_S):
            x_i = i[t]  # scalar index
            for d_idx in range(0, BLOCK_D):
                q = tl.load(base_q + d_idx * stride_qd).to(tl.float32)  # scalar
                q_vec[d_idx] = q
        # Load K block: shape [BLOCK_S, BLOCK_D]
        base_k = Kexp_ptr + b * stride_kb + h * stride_kh + (pid_s * BLOCK_S + tl.arange(0, BLOCK_S)) * stride_kj  # j is pid_s * BLOCK_S + arange
        k_mat = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
        for t in range(0, BLOCK_S):
            j = pid_s * BLOCK_S + t
            for d_idx in range(0, BLOCK_D):
                k = tl.load(Kexp_ptr + b * stride_kb + h * stride_kh + j * stride_kj + (k0 + d_idx) * stride_kd).to(tl.float32)
                k_mat[t, d_idx] = k
        # Accumulate dot: sum over BLOCK_D of q_vec[d] * k_mat[:, d]
        for d_idx in range(0, BLOCK_D):
            acc += q_vec[d_idx] * tl.sum(k_mat * 0.0 + k_mat[:, d_idx])  # multiply by SCALING inside loop? Let's do it properly
        # Apply scaling
    # Store acc to Attn
    attn_ptrs = Attn_ptr + b * stride_attnb + h * stride_attnh + i * stride_attni
    tl.store(attn_ptrs, acc, mask=(i < S))

# 8) Softmax along last dim (j) per (b, h, i) with causal mask: attn[b, h, i, j] = -inf if j <= i
# Input Attn [B, H, S, S], Output Soft [B, H, S, S]
@triton.jit
def softmax_rows_causal_kernel(
    Attn_ptr, Soft_ptr,
    B, H, S,
    stride_attnb, stride_attnh, stride_attni, stride_attnj,
    stride_sob, stride_soh, stride_soi, stride_soj,
):
    total = B * H * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (H * S)
    hs = pid % (H * S)
    h = hs // S
    i = hs % S

    base_a = Attn_ptr + b * stride_attnb + h * stride_attnh + i * stride_attni
    # Load row vector a[j]
    a = tl.zeros((S,), dtype=tl.float32)
    for j in range(0, S):
        a[j] = tl.load(base_a + j * stride_attnj).to(tl.float32)
    # Apply causal mask: set j <= i to -inf
    for j in range(0, S):
        if j <= i:
            a[j] = -float('inf')
    # Stable softmax
    a_max = tl.max(a)
    a = a - a_max
    exp_a = tl.exp(a)
    sum_a = tl.sum(exp_a)
    out = exp_a / sum_a
    base_out = Soft_ptr + b * stride_sob + h * stride_soh + i * stride_soi
    for j in range(0, S):
        tl.store(base_out + j * stride_soj, out[j])

# 9) Output projection: C[M, N] = A[M, K] @ B[N, K]^T (no bias)
# Here A is attn_output flattened [B*S, Hq*D] and B is o_proj_weight [Hq*D, Hq*D]; Output [B*S, Hq*D].
# We implement as a GEMM-like kernel.
@triton.jit
def output_proj_kernel(
    A_ptr, B_ptr, C_ptr,
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

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# ---------- ModelNew ----------

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # Ensure device and dtype
        device = hidden_states.device
        B, S_in, K = hidden_states.shape  # K = Hq*HEAD_DIM = 12288
        Hq = NUM_ATTENTION_HEADS
        Hk = NUM_KEY_VALUE_HEADS
        Hv = NUM_KEY_VALUE_HEADS  # original V has Hv = 8
        D = HEAD_DIM
        groups = NUM_KEY_VALUE_GROUPS

        # 1) Linear projections (use Triton GEMM + bias)
        # Allocate Q, K, V
        Q = torch.empty((B, S_in, Hq * D), device=device, dtype=torch.float32)
        K = torch.empty((B, S_in, Hk * D), device=device, dtype=torch.float32)
        V = torch.empty((B, S_in, Hv * D), device=device, dtype=torch.float32)

        # We'll feed hidden_states to kernel as [B*S_in, K] for A, and weights as [N, K]
        # Define grid and launch linear_gemm_bias for Q, K, V
        # For Q: A = hidden_states, B = q_proj_weight, Bias = q_proj_bias
        # For K: A = hidden_states, B = k_proj_weight, Bias = k_proj_bias
        # For V: A = hidden_states, B = v_proj_weight, Bias = v_proj_bias

        # Compute M, N, K for each
        M_q = B * S_in
        N_q = Hq * D
        K_q = K  # 12288

        M_k = B * S_in
        N_k = Hk * D
        K_k = K

        M_v = B * S_in
        N_v = Hv * D
        K_v = K

        # Bias tensors
        Bias_Q = q_proj_bias
        Bias_K = k_proj_bias
        Bias_V = v_proj_bias

        # Launch Q
        grid_q = (triton.cdiv(M_q, 64), triton.cdiv(N_q, 64))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, Bias_Q, Q,
            M_q, N_q, K_q,
            1, K,  # stride_am, stride_ak
            N_q, K,  # stride_bn, stride_bk
            S_in * Hq, D,  # stride_cm, stride_cn (we're writing [B, S_in, Hq*D])
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # Launch K
        grid_k = (triton.cdiv(M_k, 64), triton.cdiv(N_k, 64))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, Bias_K, K,
            M_k, N_k, K_k,
            1, K,
            N_k, K,
            S_in * Hk, D,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # Launch V
        grid_v = (triton.cdiv(M_v, 64), triton.cdiv(N_v, 64))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, Bias_V, V,
            M_v, N_v, K_v,
            1, K,
            N_v, K,
            S_in * Hv, D,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # 2) Reshape to heads
        # Qh: [B, S_in, Hq, D]
        Qh = Q.view(B, S_in, Hq, D)
        # Kh: [B, S_in, Hk, D]
        Kh = K.view(B, S_in, Hk, D)
        # Vh: [B, S_in, Hv, D]
        Vh = V.view(B, S_in, Hv, D)

        # 3) RMSNorm per head for Q and K
        # Allocate Q_norm and K_norm
        Qnorm = torch.empty_like(Qh, dtype=torch.float32, device=device)
        Knorm = torch.empty_like(Kh, dtype=torch.float32, device=device)

        # Launch RMSNorm for Q: each (b, s, h) row
        total_q = B * S_in * Hq
        grid_rms_q = (total_q,)
        rmsnorm_q_kernel[grid_rms_q](
            Qh, q_norm_weight, Qnorm,
            B, S_in, Hq, D,
            S_in * Hq, D, Hq, D,
            q_norm_weight.stride(0),  # stride_w = 1
            S_in, Hq, D,
            RMS_EPS
        )

        # Launch RMSNorm for K
        total_k = B * S_in * Hk
        grid_rms_k = (total_k,)
        rmsnorm_k_kernel[grid_rms_k](
            Kh, k_norm_weight, Knorm,
            B, S_in, Hk, D,
            S_in * Hk, D, Hk, D,
            k_norm_weight.stride(0),
            S_in, Hk, D,
            RMS_EPS
        )

        # 4) Rotate last half for Q and K
        Qrot = torch.empty_like(Qh, dtype=torch.float32, device=device)
        Kr = torch.empty_like(Kh, dtype=torch.float32, device=device)

        grid_rotate_q = (B * S_in * Hq,)
        rotate_q_half_kernel[grid_rotate_q](
            Qnorm, cos, sin, Qrot,
            B, S_in, Hq, D,
            S_in * Hq, D, Hq, D,
            1,  # cos/sin are scalars, stride 1
            S_in, Hq, D,
            BLOCK_D=64
        )

        grid_rotate_k = (B * S_in * Hk,)
        rotate_k_half_kernel[grid_rotate_k](
            Knorm, cos, sin, Kr,
            B, S_in, Hk, D,
            S_in * Hk, D, Hk, D,
            1,
            S_in, Hk, D,
            BLOCK_D=64
        )

        # 5) GQA expand K and V from Hk/Hv to Hq
        Kexp = torch.empty((B, Hq, S_in, D), device=device, dtype=torch.float32)
        Vexp = torch.empty((B, Hq, S_in, D), device=device, dtype=torch.float32)

        grid_expand = (B * Hq * S_in,)
        expand_k_kernel[grid_expand](
            Kh, Kexp,
            B, S_in, Hq, Hk, D,
            S_in * Hk, D, Hk, D,
            B, S_in, Hq, D,
            D,
            GROUPS
        )

        expand_v_kernel[grid_expand](
            Vh, Vexp,
            B, S_in, Hq, Hv, D,
            S_in * Hv, D, Hv, D,
            B, S_in, Hq, D,
            D,
            GROUPS
        )

        # 6) Compute attention scores: Attn [B, Hq, S_in, S_in]
        Attn = torch.empty((B, Hq, S_in, S_in), device=device, dtype=torch.float32)
        # grid over (B*Hq, tiles of S)
        grid_attn = (B * Hq, triton.cdiv(S_in, 64))
        attention_scores_kernel[grid_attn](
            Qrot, Kexp, Attn,
            B, Hq, S_in, D,
            S_in * Hq, Hq, S_in, D,
            B, Hq, S_in, D,
            SCALING,
            64, 64
        )

        # 7) Softmax along j with causal mask
        Soft = torch.empty_like(Attn, device=device, dtype=torch.float32)

        # Implement causal mask within kernel (we already wrote Attn; we can compute softmax here)
        # But since we compute attention scores in a kernel above, we call softmax_rows_causal_kernel
        # Note: we will run the kernel and overwrite Soft
        grid_softmax = (B * Hq * S_in,)
        softmax_rows_causal_kernel[grid_softmax](
            Attn, Soft,
            B, Hq, S_in,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
        )

        # 8) Compute attention output: attn_output[b, i, :] = sum_j Soft[b, i, j] * Vexp[b, i, j, :]
        attn_output = torch.empty((B, S_in, Hq * D), device=device, dtype=torch.float32)

        # We need to implement this GEMM-like without PyTorch ops. This is compute-heavy; for simplicity and correctness,
        # we can implement it as:
        # For each (b, i), loop over h_out in [0..95], compute dot over D, accumulate.


def run(*args):
    return ModelNew()(*args)
