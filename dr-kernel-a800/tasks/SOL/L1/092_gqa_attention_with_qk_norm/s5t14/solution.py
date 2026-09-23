import torch
import triton
import triton.language as tl

# Constants (same as original code)
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)
RMS_EPS = 1e-6

# Utility: ceil-div
def _ceil_div(a, b):
    return (a + b - 1) // b


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

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    # Add bias: broadcast along rows
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Triton RMSNorm over last dim (D): input [B, S, H, D] -> output [B, S, H, D]
@triton.jit
def rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    # Reduce over D to get variance
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + EPS)

    # Apply norm and optional affine weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32) * inv_rms
        # weight is scalar applied uniformly across D
        w = tl.load(W_ptr)  # scalar weight
        x = x * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, x, mask=mask)


# 3) Triton GQA expand K/V to 96 heads from 8 heads using groups: [B, Hk, S, D] -> [B, H, S, D] by repeating groups
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, S, Hk, D,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    NUM_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    total = B * S * Hk
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * Hk)
    h_k = pid % (S * Hk)

    # We will repeat Hk * NUM_GROUPS -> H, mapping h_k*g -> h
    # However we don't have H as input. We can only expand given Hk, S, D and NUM_GROUPS, but we don't know final H here.
    # Since original code repeats to NUM_ATTENTION_HEADS, we assume grid size launches enough programs.
    # Each program handles copying one Hk head into its NUM_GROUPS replicas.
    for g in range(NUM_GROUPS):
        h = h_k * NUM_GROUPS + g
        for s0 in range(0, S):
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask = offs_d < D
                x = tl.load(X_ptr + b * stride_xb + h_k * stride_xh + s0 * stride_xs + offs_d * stride_xd, mask=mask, other=0.0)
                tl.store(Y_ptr + b * stride_yb + h * stride_yh + s0 * stride_ys + offs_d * stride_yd, x, mask=mask)


# 4) Triton matmul for attention scores: S[b,h,i,j] = Qrot[b,h,i,:] @ Krot[b,h,j,:]^T
# We implement S[b,h,i,j] using two 2D tiles: we set grid over (B*H, tiles_i, tiles_j)
@triton.jit
def attn_scores_matmul_kernel(
    Q_ptr, K_ptr, S_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_si, stride_sj,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    offs_i = pid_i * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_j = pid_j * BLOCK_S + tl.arange(0, BLOCK_S)

    acc = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)

    # Loop over D (head dim) in chunks
    for d0 in range(0, D, BLOCK_D):
        q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_i[:, None] * stride_qs + (d0 + tl.arange(0, BLOCK_D))[None, :] * stride_qd  # [BS, BD]
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_j[None, :] * stride_ks + (d0 + tl.arange(0, BLOCK_D))[:, None] * stride_kd  # [BD, BS]

        mask_q = (offs_i[:, None] < S) & ((d0 + tl.arange(0, BLOCK_D))[None, :] < D)
        mask_k = (offs_j[None, :] < S) & ((d0 + tl.arange(0, BLOCK_D))[:, None] < D)

        q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)
        k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.dot(q, k)  # [BS, BS]

    # Store to S[b,h,i,j]
    s_ptrs = S_ptr + b * stride_sb + h * stride_sh + offs_i[:, None] * stride_si + offs_j[None, :] * stride_sj
    mask_s = (offs_i[:, None] < S) & (offs_j[None, :] < S)
    tl.store(s_ptrs, acc, mask=mask_s)


# 5) Triton softmax over last dim (j) for each (b,h,i): S[b,h,i,:] = softmax(S[b,h,i,:])
@triton.jit
def softmax_rows_kernel(
    S_ptr, S_out_ptr,
    B, H, S,
    stride_sb, stride_sh, stride_si, stride_sj,
    BLOCK_S: tl.constexpr,
):
    total = B * H * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Row vector pointers for i-th row across j in [0..S)
    # We need to load row, compute max, exp, sum, and write normalized values
    row_max = -float('inf')
    # Compute max across j
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask = offs_j < S
        s_ptrs = S_ptr + b * stride_sb + h * stride_sh + i * stride_si + offs_j * stride_sj
        s = tl.load(s_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(s, axis=0))

    # Compute exp and sum
    row_sum = 0.0
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask = offs_j < S
        s_ptrs = S_ptr + b * stride_sb + h * stride_sh + i * stride_si + offs_j * stride_sj
        s = tl.load(s_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        e = tl.exp(s - row_max)
        tl.store(S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + offs_j * stride_sj, e, mask=mask)
        row_sum += tl.sum(e, axis=0)

    # Normalize
    inv_sum = 1.0 / row_sum
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask = offs_j < S
        e_ptrs = S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + offs_j * stride_sj
        e = tl.load(e_ptrs, mask=mask, other=0.0).to(tl.float32)
        e = e * inv_sum
        tl.store(S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + offs_j * stride_sj, e, mask=mask)


# 6) Triton matmul to compute attention output: Out[b,h,:] = S[b,h,:] @ V[b,h,:]^T
@triton.jit
def attn_output_matmul_kernel(
    S_ptr, V_ptr, Out_ptr,
    B, H, S, D,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    acc = tl.zeros((D,), dtype=tl.float32)

    for i in range(0, S):
        s_ptrs = S_ptr + b * stride_sb + h * stride_sh + i * stride_si + tl.arange(0, S) * stride_sj
        s = tl.load(s_ptrs, mask=True, other=0.0).to(tl.float32)  # [S]
        for j in range(0, S):
            # Gather v[j, :] -> [D]
            v_ptrs = V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + tl.arange(0, D) * stride_vd
            v = tl.load(v_ptrs, mask=True, other=0.0).to(tl.float32)  # [D]
            acc += s[j] * v
        # Store acc to Out[b,h,i,:]
        out_ptrs = Out_ptr + b * stride_ob + h * stride_oh + i * stride_os + tl.arange(0, D) * stride_od
        tl.store(out_ptrs, acc, mask=True)


# 7) Triton output projection: Out2[B*S, H*D] @ o_proj_weight^T + bias -> [B, S, H*D]
# We implement GEMM without bias here; bias can be added in host or by passing a separate bias
@triton.jit
def linear_output_nobias_kernel(
    In_ptr, W_ptr, Out_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wm, stride_wk,  # W is [M, K] for output projection
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
        in_ptrs = In_ptr + (offs_m[:, None] * stride_im + (k + offs_k)[None, :] * stride_ik)  # [BM, BK]
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wm + (k + offs_k)[:, None] * stride_wk)   # [BK, BN]

        mask_in = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        mask_w = (offs_n[None, :] < N) & ((k + offs_k)[:, None] < K)

        in_vals = tl.load(in_ptrs, mask=mask_in, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)

        acc += tl.dot(in_vals, w_vals)

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        """
        hidden_states: [B, S, D], q_proj_weight: [H*D, D], k/v/o_proj_weight: [D, H*D], norm weights: [D]
        cos, sin: [D]
        Returns output tensor [B, S, H*D] where H=NUM_ATTENTION_HEADS, D=HEAD_DIM.
        """

        assert hidden_states.dim() == 3, "hidden_states must be [B, S, D]"
        assert hidden_states.is_cuda, "ModelNew requires CUDA device for Triton kernels"

        B, S, D = hidden_states.shape
        device = hidden_states.device

        # 1) Linear projections using Triton: Q, K, V
        hidden_f32 = hidden_states.to(torch.float32)
        Mq = B * S

        # Q = hidden @ q_proj_weight^T + q_proj_bias
        Q = torch.empty((B, S, D), device=device, dtype=torch.float32)
        Aq = hidden_f32.reshape(Mq, D).contiguous()
        Nq = NUM_ATTENTION_HEADS * D
        Bq = q_proj_weight  # [Nq, D]
        Bias_q = q_proj_bias if q_proj_bias is not None else torch.zeros((Nq,), device=device, dtype=torch.float32)
        grid_q = (_ceil_div(Mq, 128), _ceil_div(Nq, 128))
        linear_gemm_bias_kernel[grid_q](
            Aq, Bq, Bias_q, Q.reshape(Mq, Nq),
            Mq, Nq, D,
            Aq.stride(0), Aq.stride(1),
            Bq.stride(0), Bq.stride(1),
            Q.reshape(Mq, Nq).stride(0), Q.reshape(Mq, Nq).stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )
        Q = Q.view(B, S, D)

        # K = hidden @ k_proj_weight^T + k_proj_bias
        K = torch.empty((B, S, D), device=device, dtype=torch.float32)
        Ak = hidden_f32.reshape(Mq, D).contiguous()
        Nk = NUM_KEY_VALUE_HEADS * D
        Bk = k_proj_weight  # [Nk, D]
        Bias_k = k_proj_bias if k_proj_bias is not None else torch.zeros((Nk,), device=device, dtype=torch.float32)
        grid_k = (_ceil_div(Mq, 128), _ceil_div(Nk, 128))
        linear_gemm_bias_kernel[grid_k](
            Ak, Bk, Bias_k, K.reshape(Mq, Nk),
            Mq, Nk, D,
            Ak.stride(0), Ak.stride(1),
            Bk.stride(0), Bk.stride(1),
            K.reshape(Mq, Nk).stride(0), K.reshape(Mq, Nk).stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )
        K = K.view(B, S, D)

        # V = hidden @ v_proj_weight^T + v_proj_bias
        V = torch.empty((B, S, D), device=device, dtype=torch.float32)
        Av = hidden_f32.reshape(Mq, D).contiguous()
        Nv = NUM_KEY_VALUE_HEADS * D
        Bv = v_proj_weight  # [Nv, D]
        Bias_v = v_proj_bias if v_proj_bias is not None else torch.zeros((Nv,), device=device, dtype=torch.float32)
        grid_v = (_ceil_div(Mq, 128), _ceil_div(Nv, 128))
        linear_gemm_bias_kernel[grid_v](
            Av, Bv, Bias_v, V.reshape(Mq, Nv),
            Mq, Nv, D,
            Av.stride(0), Av.stride(1),
            Bv.stride(0), Bv.stride(1),
            V.reshape(Mq, Nv).stride(0), V.reshape(Mq, Nv).stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )
        V = V.view(B, S, D)

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q, device=device, dtype=torch.float32)
        K_norm = torch.empty_like(K, device=device, dtype=torch.float32)

        grid_rms = (B * S * NUM_ATTENTION_HEADS, _ceil_div(D, 128))
        rms_norm_kernel[grid_rms](
            Q.view(B, S, NUM_ATTENTION_HEADS, D),
            q_norm_weight, Q_norm.view(B, S, NUM_ATTENTION_HEADS, D),
            B, S, NUM_ATTENTION_HEADS, D,
            Q.view(B, S, NUM_ATTENTION_HEADS, D).stride(0), Q.view(B, S, NUM_ATTENTION_HEADS, D).stride(1),
            Q.view(B, S, NUM_ATTENTION_HEADS, D).stride(2), Q.view(B, S, NUM_ATTENTION_HEADS, D).stride(3),
            Q_norm.view(B, S, NUM_ATTENTION_HEADS, D).stride(0), Q_norm.view(B, S, NUM_ATTENTION_HEADS, D).stride(1),
            Q_norm.view(B, S, NUM_ATTENTION_HEADS, D).stride(2), Q_norm.view(B, S, NUM_ATTENTION_HEADS, D).stride(3),
            EPS=RMS_EPS, BLOCK_D=128
        )

        grid_rms_k = (B * S * NUM_KEY_VALUE_HEADS, _ceil_div(D, 128))
        rms_norm_kernel[grid_rms_k](
            K.view(B, S, NUM_KEY_VALUE_HEADS, D),
            k_norm_weight, K_norm.view(B, S, NUM_KEY_VALUE_HEADS, D),
            B, S, NUM_KEY_VALUE_HEADS, D,
            K.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(0), K.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(1),
            K.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(2), K.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(3),
            K_norm.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(0), K_norm.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(1),
            K_norm.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(2), K_norm.view(B, S, NUM_KEY_VALUE_HEADS, D).stride(3),
            EPS=RMS_EPS, BLOCK_D=128
        )

        # 3) Rotate half for Q and K (RoPE-like)
        # Prepare rotated Q
        Q_rot = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        Q_n = Q_norm.reshape(B, S, NUM_ATTENTION_HEADS, D)
        # Launch rotate for Q on all (B,S,H) elements
        for b in range(B):
            for s in range(S):
                for h in range(NUM_ATTENTION_HEADS):
                    q = Q_n[b, s, h]  # [D]
                    # cat (-q2, q1) with q1=first 64, q2=second 64
                    q1 = q[:64]
                    q2 = q[64:]
                    q_rot_half = torch.empty_like(q)
                    q_rot_half[:64] = -q2
                    q_rot_half[64:] = q1
                    # Apply rotation with cos/sin
                    # Broadcast to [D]
                    cos_vec = cos.to(torch.float32)
                    sin_vec = sin.to(torch.float32)
                    q_out = q * cos_vec + q_rot_half * sin_vec
                    Q_rot[b, s, h] = q_out

        # Prepare rotated K
        K_rot = torch.empty_like(K_norm, device=device, dtype=torch.float32)
        K_n = K_norm.reshape(B, S, NUM_KEY_VALUE_HEADS, D)
        for b in range(B):
            for s in range(S):
                for h in range(NUM_KEY_VALUE_HEADS):
                    k = K_n[b, s, h]  # [D]
                    k1 = k[:64]
                    k2 = k[64:]
                    k_rot_half = torch.empty_like(k)
                    k_rot_half[:64] = -k2
                    k_rot_half[64:] = k1
                    cos_vec = cos.to(torch.float32)
                    sin_vec = sin.to(torch.float32)
                    k_out = k * cos_vec + k_rot_half * sin_vec
                    # We need to place k_out into expanded K_rot[b,h] positions
                    # Note: K_rot shape is [B,S,NUM_KEY_VALUE_HEADS,D]
                    K_rot[b, s, h] = k_out

        # 4) GQA expand K and V to 96 heads
        K_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, D), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, D), device=device, dtype=torch.float32)

        # Launch gqa_expand_kernel: grid over B*S*Hk
        grid_gqa = (B * S * NUM_KEY_VALUE_HEADS,)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, NUM_KEY_VALUE_HEADS, D,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            NUM_KEY_VALUE_GROUPS, BLOCK_D=128
        )

        grid_gqa_v = (B * S * NUM_KEY_VALUE_HEADS,)
        gqa_expand_kernel[grid_gqa_v](
            V, V_expanded,
            B, S, NUM_KEY_VALUE_HEADS, D,
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            NUM_KEY_VALUE_GROUPS, BLOCK_D=128
        )

        # 5) Compute attention scores S[b,h,S,S] via Triton
        S_scores = torch.empty((B, NUM_ATTENTION_HEADS, S, S), device=device, dtype=torch.float32)
        grid_attn = (B * NUM_ATTENTION_HEADS, _ceil_div(S, 128), _ceil_div(S, 128))
        attn_scores_matmul_kernel[grid_attn](
            Q_rot.reshape(B, NUM_ATTENTION_HEADS, S, D),
            K_expanded.reshape(B, NUM_ATTENTION_HEADS, S, D),
            S_scores,
            B, NUM_ATTENTION_HEADS, S, D,
            Q_rot.reshape(B, NUM_ATTENTION_HEADS, S, D).stride(0), Q_rot.reshape(B, NUM_ATTENTION_HEADS, S, D).stride(1),
            Q_rot.reshape(B, NUM_ATTENTION_HEADS, S, D).stride(2), Q_rot.reshape(B, NUM_ATTENTION_HEADS, S, D).stride(3),
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3),
            BLOCK_S=128, BLOCK_D=64
        )

        # 6) Apply causal masking in Triton: set S[i,j] = -inf if j < i
        # We can implement causal mask inside softmax_rows_kernel by loading S[b,h,i,j] as -inf when j < i.
        # But softmax_rows_kernel expects the entire row; better pre-fill S_scores with -inf and then overwrite valid entries.
        # Triton kernels don't allow writing into S_scores and reading simultaneously; we can fuse causal mask by not using softmax_rows here.
        # Instead, we compute softmax in attn_scores_matmul_kernel? Not possible. So we pre-mask S_scores.
        # Create a temporary S_scores2 with causal -inf before softmax.
        S_scores2 = torch.empty_like(S_scores, device=device, dtype=torch.float32)
        for b in range(B):
            for h in range(NUM_ATTENTION_HEADS):
                for i in range(S):
                    for j in range(S):
                        if j < i:
                            # Set to -inf
                            S_scores2[b, h, i, j] = -float('inf')

        # Now run softmax_rows_kernel on S_scores2 (it normalizes each row i across j)
        grid_softmax = (B * NUM_ATTENTION_HEADS * S,)
        softmax_rows_kernel[grid_softmax](
            S_scores2, S_scores2,
            B, NUM_ATTENTION_HEADS, S,
            S_scores2.stride(0), S_scores2.stride(1), S_scores2.stride(2), S_scores2.stride(3),
            BLOCK_S=128
        )

        # 7) Compute attention output: Out[b,h] = softmax[b,h] @ V_expanded[b,h]
        attn_output = torch.empty((B, NUM_ATTENTION_HEADS, S, D), device=device, dtype=torch.float32)

        # We need a Triton kernel that performs a row-wise matmul of S[b,h,i,:] with V_expanded[b,h,j,:] to produce Out[b,h,i,:].
        # Implement a simple loop kernel (fine for moderate S and D):
        for b in range(B):
            for h in range(NUM_ATTENTION_HEADS):
                for i in range(S):
                    acc = tl.zeros((D,), dtype=tl.float32)
                    for j in range(0, S):
                        s = S_scores2[b, h, i, j]
                        # Load V_expanded[b,h,j,:]
                        v_ptrs = V_expanded[b, h, j] + tl.arange(0, D) * V_expanded.stride(3)
                        v = tl.load(v_ptrs, mask=True, other=0.0).to(tl.float32)
                        acc += s * v
                    out_ptrs = attn_output[b, h, i] + tl.arange(0, D) * attn_output[b, h, i].stride(1)
                    tl.store(out_ptrs, acc, mask=True)

        # 8) Transpose and reshape to [B, S, H*D]
        attn_output_t = attn_output.transpose(1, 2).contiguous()  # [B, S, 96*128]

        # 9) Output projection: [B*S, H*D] @ o_proj_weight^T -> [B*S, H*D]
        Ho = NUM_ATTENTION_HEADS * D
        Out_flat = attn_output_t.reshape(B * S, Ho)
        Out_proj = torch.empty((B * S, Ho), device=device, dtype=torch.float32)
        grid_out = (_ceil_div(B * S, 128), _ceil_div(Ho, 128))
        linear_output_nobias_kernel[grid_out](
            Out_flat, o_proj_weight, Out_proj,
            B * S, Ho, D,
            Out_flat.stride(0), Out_flat.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out_proj.stride(0), Out_proj.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )

        # Reshape back to [B, S, H*D]
        Out_proj = Out_proj.view(B, S, Ho)

        return Out_proj


def run(*args):
    return ModelNew()(*args)
