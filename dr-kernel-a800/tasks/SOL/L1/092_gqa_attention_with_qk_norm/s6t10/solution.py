import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

# 1) Batched GEMM (no bias): C[M, N] = A[M, K] @ B[K, N], where B is weight^T
@triton.jit
def triton_batched_gemm_no_bias(
    A_ptr,        # *fp32, [M, K]
    B_ptr,        # *fp32, [K, N]
    C_ptr,        # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,   # A strides
    stride_bk, stride_bn,   # B strides
    stride_cm, stride_cn,   # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

# 2) RMSNorm per head across head_dim: x_hat = weight * x / sqrt(mean(x^2) + eps)
@triton.jit
def triton_rms_norm(
    X_ptr,         # *fp32, [M, D] where M = B * num_heads
    Weight_ptr,    # *fp32, [D] or scalar if not per-dim
    Out_ptr,       # *fp32, [M, D]
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_outm, stride_outd,
    eps,            # float32 scalar
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    # compute mean of squares across D
    sum_sq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + offs_d
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xd, mask=offs < D, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    # normalize and apply weight
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + offs_d
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xd, mask=offs < D, other=0.0)
        w = tl.load(Weight_ptr + offs, mask=offs < D, other=1.0)
        y = x * inv_rms * w
        tl.store(Out_ptr + m * stride_outm + offs * stride_outd, y, mask=offs < D)

# 3) Rotary Position Embedding rotate: apply to [M, D] tensor (Q or K per head)
@triton.jit
def triton_rotate_pe(
    X_ptr,         # *fp32, [M, D], input to rotate (e.g., Q or K)
    Cos_ptr,       # *fp32, [D]
    Sin_ptr,       # *fp32, [D]
    Out_ptr,       # *fp32, [M, D], output rotated
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_outm, stride_outd,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + offs_d
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xd, mask=offs < D, other=0.0)
        cosv = tl.load(Cos_ptr + offs, mask=offs < D, other=1.0)
        sinv = tl.load(Sin_ptr + offs, mask=offs < D, other=0.0)
        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotated = tl.cos(x1 * cosv[:half] + x2 * sinv[half:])  # rotate by half dims using cos/sin
        # Note: simple formula assuming half=64 for 128-D; with D=128 and fixed config, this matches original.
        # rotated = cos * x1 - sin * x2 (see original code)
        # However, the original uses cos/sin per position; the code computes:
        # q1, q2 = query[..., :64], query[..., 64:], and updates as cat((-q2, q1), dim=-1)
        # Then query = query * cos + rotated * sin. We implement that directly below.
        # To match original exactly, we implement the original cat + multiply using these cos/sin.
        # But Triton here does not have tensor cat ops across dimensions; thus we implement using vector formulas.
        # Here we follow the original logic: rotate_half = cat((-x2, x1)), then out = x * cos + rotate_half * sin.
        # For simplicity, we recompute the rotation as in original: build rotate_half explicitly with indices.
        # We'll reconstruct the rotation using half and full vectors:
        # x1 = x[ :half], x2 = x[half : ]
        # rotated_half = -x2, x1'
        # Then out = x * cos + rotated_half * sin.
        # Implement using vector ops:
        # rotated = x * cos + (-x2) * sin + x1' * sin ? Correction: rotated_half is (-x2, x1), so out =
        # out1 = x1 * cos + x2 * sin
        # out2 = -x2 * cos + x1 * sin
        # But we don't have separate x1/x2 here without split. For D=128, we can do:
        out1 = x[:half] * cosv[:half] + x[half:] * sinv[half:]
        out2 = -x[half:] * cosv[half:] + x[:half] * sinv[:half]
        out = tl.zeros((D,), dtype=tl.float32)
        out[:half] = out1
        out[half:] = out2
        tl.store(Out_ptr + m * stride_outm + offs * stride_outd, out, mask=offs < D)

# 4) Grouped Query Attention: expand num_key_value_heads to num_attention_heads by repeating groups
#    Inputs: X [B, num_key_value_heads, S, D]  (KV per head after RMSNorm)
#    Output: Out [B, num_attention_heads, S, D] (repeat each of 8 heads 12 times)
@triton.jit
def triton_gqa_expand(
    X_ptr,         # *fp32, [B, G, S, D]
    Out_ptr,       # *fp32, [B, H, S, D]
    B: tl.constexpr, G: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xg, stride_xs, stride_xd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)  # h in [0, H)
    s = tl.program_id(2)
    # Determine group idx g = h % G
    g = h % G
    offs_d = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + offs_d
        x = tl.load(
            X_ptr + b * stride_xb + g * stride_xg + s * stride_xs + offs * stride_xd,
            mask=offs < D,
            other=0.0
        )
        tl.store(
            Out_ptr + b * stride_ob + h * stride_oh + s * stride_os + offs * stride_od,
            x,
            mask=offs < D
        )

# 5) Compute attention scores: S[M, N] = Q[M, D] @ K^T[N, D], M = B * num_attention_heads, N = S
@triton.jit
def triton_score_matmul(
    Q_ptr,         # *fp32, [M, D]
    K_ptr,         # *fp32, [N, D]
    S_ptr,         # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0
        )  # [BLOCK_M, BLOCK_D]
        k = tl.load(
            K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0
        )  # [BLOCK_N, BLOCK_D]
        acc += tl.dot(q, tl.trans(k))
    acc = acc * SCALE
    tl.store(
        S_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

# 6) Row-wise softmax: In_ptr [M, N], Out_ptr [M, N]
@triton.jit
def triton_softmax_rows(
    In_ptr,        # *fp32, [M, N]
    Out_ptr,       # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    # Load row
    row = tl.load(In_ptr + pid_m * stride_im + offs_n * stride_in, mask=offs_n < N, other=-1e20)
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    row_sum = tl.sum(exp_row, axis=0)
    out_row = exp_row / row_sum
    tl.store(Out_ptr + pid_m * stride_om + offs_n * stride_on, out_row, mask=offs_n < N)

# 7) Final attention output: Out[M, N] = Softmax(S)[M, N] @ V[N, D]
@triton.jit
def triton_final_output(
    S_ptr,         # *fp32, [M, N]
    V_ptr,         # *fp32, [N, D]
    Out_ptr,       # *fp32, [M, D]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_sm, stride_sn,
    stride_vm, stride_vd,  # note: V is [N, D], we need (row=N, col=D)
    stride_om, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_offs = n0 + offs_n
        sm = tl.load(S_ptr + pid_m * stride_sm + n_offs * stride_sn, mask=n_offs < N, other=0.0)  # [BLOCK_N]
        v = tl.load(V_ptr + n_offs[:, None] * stride_vm + offs_d[None, :] * stride_vd, mask=(n_offs[:, None] < N) & (offs_d[None, :] < D), other=0.0)  # [BLOCK_N, BLOCK_D]
        acc += tl.sum(sm[:, None] * v, axis=0)
    tl.store(Out_ptr + pid_m * stride_om + offs_d * stride_od, acc, mask=offs_d < D)


# =========================
# ModelNew: TRITON-ONLY forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 1100, num_attention_heads: int = 96, num_key_value_heads: int = 8, head_dim: int = 128, num_key_value_groups: int = 12, rms_norm_eps: float = 1e-6):
        super().__init__()
        # Keep config for Triton kernels
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_key_value_groups
        self.rms_norm_eps = rms_norm_eps
        # We will receive weights in forward; not storing them here to keep forward minimal and Triton-only.

    def forward(
        self,
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
        sin: torch.Tensor,
    ):
        # Shapes from input hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        assert H == self.num_attention_heads * self.head_dim, "hidden_dim mismatch with num_attention_heads and head_dim"
        # We only use weights for Q/K/V/o_proj in Triton GEMM. Keep everything in fp32.
        device = hidden_states.device

        # 1) Dense projection: Q = hidden_states @ q_proj_weight^T
        # Reshape hidden_states to [M, K] where M = B * num_attention_heads, K = head_dim
        # For Q: M = B * num_attention_heads, K = head_dim, N = hidden_dim
        M_Q = B * self.num_attention_heads
        K = self.head_dim
        N_Q = H  # equals num_attention_heads * head_dim
        A_Q = hidden_states.view(M_Q, K).contiguous().to(torch.float32)
        Bt_Q = q_proj_weight.t().contiguous().to(torch.float32)  # [head_dim, hidden_dim]
        Q = torch.empty((M_Q, N_Q), device=device, dtype=torch.float32)
        grid_gemm_q = (triton.cdiv(M_Q, 64), triton.cdiv(N_Q, 64))
        triton_batched_gemm_no_bias[grid_gemm_q](
            A_Q, Bt_Q, Q,
            M_Q, N_Q, K,
            A_Q.stride(0), A_Q.stride(1),
            Bt_Q.stride(0), Bt_Q.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        # Reshape Q back to [B, num_attention_heads, head_dim]
        Q = Q.view(B, self.num_attention_heads, self.head_dim)

        # 2) RMSNorm for Q
        Q_norm_weight = q_norm_weight.to(torch.float32)
        Q_norm = torch.empty_like(Q, dtype=torch.float32)
        grid_rms_q = (M_Q, 1)
        triton_rms_norm[grid_rms_q](
            Q.view(M_Q, self.head_dim), Q_norm_weight, Q_norm.view(M_Q, self.head_dim),
            M_Q, self.head_dim,
            Q.view(M_Q, self.head_dim).stride(0), Q.view(M_Q, self.head_dim).stride(1),
            Q_norm.view(M_Q, self.head_dim).stride(0), Q_norm.view(M_Q, self.head_dim).stride(1),
            self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )
        Q_norm = Q_norm.view(B, self.num_attention_heads, self.head_dim)

        # 3) RotPE for Q
        cos_q = cos[:self.head_dim].to(torch.float32)
        sin_q = sin[:self.head_dim].to(torch.float32)
        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32)
        grid_rot_q = (B * self.num_attention_heads, 1)
        triton_rotate_pe[grid_rot_q](
            Q_norm.view(B * self.num_attention_heads, self.head_dim),
            cos_q, sin_q,
            Q_rot.view(B * self.num_attention_heads, self.head_dim),
            B * self.num_attention_heads, self.head_dim,
            Q_norm.view(B * self.num_attention_heads, self.head_dim).stride(0), Q_norm.view(B * self.num_attention_heads, self.head_dim).stride(1),
            Q_rot.view(B * self.num_attention_heads, self.head_dim).stride(0), Q_rot.view(B * self.num_attention_heads, self.head_dim).stride(1),
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )
        Q_rot = Q_rot.view(B, self.num_attention_heads, self.head_dim)

        # 4) K projection and RMSNorm
        # K: same as Q logic
        M_K = B * self.num_attention_heads  # but attention uses key_value_heads for K/V
        # We need K per key head: first compute K using k_proj_weight (hidden_dim -> hidden_dim)
        # Reshape to [M, K] where M = B * num_key_value_heads, K = head_dim, N = hidden_dim
        B_K = hidden_states.view(B * self.num_key_value_heads, self.head_dim).contiguous().to(torch.float32)
        Bt_K = k_proj_weight.t().contiguous().to(torch.float32)  # [head_dim, hidden_dim]
        K_t = torch.empty((B * self.num_key_value_heads, H), device=device, dtype=torch.float32)
        grid_gemm_k = (triton.cdiv(B * self.num_key_value_heads, 64), triton.cdiv(H, 64))
        triton_batched_gemm_no_bias[grid_gemm_k](
            B_K, Bt_K, K_t,
            B * self.num_key_value_heads, H, self.head_dim,
            B_K.stride(0), B_K.stride(1),
            Bt_K.stride(0), Bt_K.stride(1),
            K_t.stride(0), K_t.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )
        # Reshape to [B, num_key_value_heads, head_dim]
        K_t = K_t.view(B, self.num_key_value_heads, self.head_dim)

        # RMSNorm for K
        K_norm_weight = k_norm_weight.to(torch.float32)
        K_norm = torch.empty_like(K_t, dtype=torch.float32)
        grid_rms_k = (B * self.num_key_value_heads, 1)
        triton_rms_norm[grid_rms_k](
            K_t.view(B * self.num_key_value_heads, self.head_dim),
            K_norm_weight,
            K_norm.view(B * self.num_key_value_heads, self.head_dim),
            B * self.num_key_value_heads, self.head_dim,
            K_t.view(B * self.num_key_value_heads, self.head_dim).stride(0), K_t.view(B * self.num_key_value_heads, self.head_dim).stride(1),
            K_norm.view(B * self.num_key_value_heads, self.head_dim).stride(0), K_norm.view(B * self.num_key_value_heads, self.head_dim).stride(1),
            self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )
        K_norm = K_norm.view(B, self.num_key_value_heads, self.head_dim)

        # RotPE for K
        cos_k = cos[:self.head_dim].to(torch.float32)
        sin_k = sin[:self.head_dim].to(torch.float32)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32)
        grid_rot_k = (B * self.num_key_value_heads, 1)
        triton_rotate_pe[grid_rot_k](
            K_norm.view(B * self.num_key_value_heads, self.head_dim),
            cos_k, sin_k,
            K_rot.view(B * self.num_key_value_heads, self.head_dim),
            B * self.num_key_value_heads, self.head_dim,
            K_norm.view(B * self.num_key_value_heads, self.head_dim).stride(0), K_norm.view(B * self.num_key_value_heads, self.head_dim).stride(1),
            K_rot.view(B * self.num_key_value_heads, self.head_dim).stride(0), K_rot.view(B * self.num_key_value_heads, self.head_dim).stride(1),
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )
        K_rot = K_rot.view(B, self.num_key_value_heads, self.head_dim)

        # 5) V projection (no bias)
        # Reshape hidden_states to [B * num_key_value_heads, head_dim]
        B_V = hidden_states.view(B * self.num_key_value_heads, self.head_dim).contiguous().to(torch.float32)
        Bt_V = v_proj_weight.t().contiguous().to(torch.float32)  # [head_dim, hidden_dim]
        V = torch.empty((B * self.num_key_value_heads, H), device=device, dtype=torch.float32)
        grid_gemm_v = (triton.cdiv(B * self.num_key_value_heads, 64), triton.cdiv(H, 64))
        triton_batched_gemm_no_bias[grid_gemm_v](
            B_V, Bt_V, V,
            B * self.num_key_value_heads, H, self.head_dim,
            B_V.stride(0), B_V.stride(1),
            Bt_V.stride(0), Bt_V.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )
        V = V.view(B, self.num_key_value_heads, self.head_dim)

        # 6) Grouped Query Attention: expand K_rot/V to 96 heads (repeat each of 8 heads 12 times)
        # We need OutK [B, 96, S, D] and OutV [B, 96, S, D]
        OutK = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=torch.float32)
        OutV = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=torch.float32)
        # Launch for K and V separately
        grid_gqa_k = (B, self.num_key_value_heads, S)
        triton_gqa_expand[grid_gqa_k](
            K_rot, OutK,
            B, self.num_key_value_heads, S, self.head_dim,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            OutK.stride(0), OutK.stride(1), OutK.stride(2), OutK.stride(3),
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )
        grid_gqa_v = (B, self.num_key_value_heads, S)
        triton_gqa_expand[grid_gqa_v](
            V, OutV,
            B, self.num_key_value_heads, S, self.head_dim,
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            OutV.stride(0), OutV.stride(1), OutV.stride(2), OutV.stride(3),
            BLOCK_D=128,
            num_warps=2, num_stages=2
        )

        # 7) Compute attention scores: S[B*num_attention_heads, S] = Q_rot[B*num_attention_heads, D] @ OutK[B*num_attention_heads, S, D]^T
        # Q_rot [B, num_attention_heads, D], OutK [B, num_attention_heads, S, D]
        # Flatten Q_rot to [M, D], OutK to [N, D]
        M_score = B * self.num_attention_heads
        N_score = S
        D_score = self.head_dim
        Q_flat = Q_rot.view(M_score, D_score).contiguous()
        OutK_flat = OutK.view(M_score, N_score, D_score).contiguous()
        S_attn = torch.empty((M_score, N_score), device=device, dtype=torch.float32)
        grid_score = (triton.cdiv(M_score, 64), triton.cdiv(N_score, 64))
        # We need OutK's [N, D]; reshape as [N_score, D_score]
        OutK_ND = OutK_flat[:, :, :].contiguous()  # [M, N, D] -> view [N, D] for each row? We'll reconstruct by selecting rows
        # Instead, construct OutK_ND as [N_score, D_score]: for each m, we take OutK[m, :, :] across S
        OutK_ND = torch.empty((N_score, D_score), device=device, dtype=torch.float32)
        # But we cannot read OutK_flat directly into [N_score, D_score]; instead, we will pass OutK as [M, N, D] and read per m in Triton.
        # To do that, we pass pointers and let Triton compute per m row. So we keep OutK as [M, N, D] and index per m in kernel.
        # However, Triton kernel expects [N, D]. We can create OutK_ND by transposing and selecting rows per m.
        # For simplicity, we'll create OutK_ND[m, :] = OutK_flat[m, :, :] by slicing:
        for m in range(M_score):
            OutK_ND[m, :] = OutK_flat[m, :, :]
        triton_score_matmul[grid_score](
            Q_flat, OutK_ND, S_attn,
            M_score, N_score, D_score,
            Q_flat.stride(0), Q_flat.stride(1),
            OutK_ND.stride(0), OutK_ND.stride(1),
            S_attn.stride(0), S_attn.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_D=64,
            SCALE=1.0 / (self.head_dim ** 0.5),
            num_warps=4, num_stages=2
        )

        # 8) Softmax over sequence dimension per row
        S_attn_soft = torch.empty_like(S_attn, dtype=torch.float32)
        grid_softmax = (M_score, 1)
        triton_softmax_rows[grid_softmax](
            S_attn, S_attn_soft,
            M_score, N_score,
            S_attn.stride(0), S_attn.stride(1),
            S_attn_soft.stride(0), S_attn_soft.stride(1),
            BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # 9) Final output: Out[B*num_attention_heads, D] = Softmax(S_attn)[M, N] @ OutV[B*num_attention_heads, S, D]
        # We need OutV in [M, N, D]; construct as OutV_mat[M, D] = OutV[m, :, :] for each m across N=S
        OutV_mat = torch.empty((M_score, D_score), device=device, dtype=torch.float32)
        for m in range(M_score):
            OutV_mat[m, :] = OutV_flat[m, :, :]
        Out_attn = torch.empty((M_score, D_score), device=device, dtype=torch.float32)
        grid_final = (M_score, 1)
        triton_final_output[grid_final](
            S_attn_soft, OutV_mat,
            Out_attn,
            M_score, N_score, D_score,
            S_attn_soft.stride(0), S_attn_soft.stride(1),
            OutV_mat.stride(0), OutV_mat.stride(1),
            Out_attn.stride(0), Out_attn.stride(1),
            BLOCK_N=128, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 10) Reshape and output projection (no bias): [B, S, num_attention_heads * head_dim] -> [B, S, hidden_dim]
        Out_attn = Out_attn.view(B, self.num_attention_heads, S, self.head_dim)
        # Reshape [B, num_attention_heads, S, head_dim] to [B*S*num_attention_heads, head_dim]
        M_proj = B * S * self.num_attention_heads
        attn_flat = Out_attn.reshape(M_proj, self.head_dim).contiguous()
        # Output projection: attn_flat @ o_proj_weight^T
        Bt_o = o_proj_weight.t().contiguous().to(torch.float32)  # [hidden_dim, num_attention_heads*head_dim]
        output = torch.empty((M_proj, H), device=device, dtype=torch.float32)
        grid_o = (triton.cdiv(M_proj, 64), triton.cdiv(H, 64))
        triton_batched_gemm_no_bias[grid_o](
            attn_flat, Bt_o, output,
            M_proj, H, self.head_dim,
            attn_flat.stride(0), attn_flat.stride(1),
            Bt_o.stride(0), Bt_o.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
