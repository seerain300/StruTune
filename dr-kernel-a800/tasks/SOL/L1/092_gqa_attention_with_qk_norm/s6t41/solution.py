import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

# 1) Batched GEMM no bias: C[M, N] = A[M, K] @ B[K, N], where B is weight^T
@triton.jit
def triton_batched_gemm_no_bias(
    A_ptr,        # *fp32, [M, K]
    B_ptr,        # *fp32, [K, N]
    C_ptr,        # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,   # A strides (row, col)
    stride_bk, stride_bn,   # B strides (row=K, col=N)
    stride_cm, stride_cn,   # C strides (row, col)
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


# 2) RMSNorm per token, per head across head_dim (normalize and scale)
@triton.jit
def triton_rmsnorm(
    X_ptr,        # *fp32, [M, D] where M = B * num_attention_heads
    Weight_ptr,   # *fp32, [D]
    Y_ptr,        # *fp32, [M, D]
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_w,
    stride_ym, stride_yd,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    # compute sum of squares across D
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(sumsq / D + eps)

    # normalize and scale
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        w = tl.load(Weight_ptr + offs_d * stride_w, mask=offs_d < D, other=1.0)
        y = x * inv * w
        tl.store(Y_ptr + pid * stride_ym + offs_d * stride_yd, y, mask=offs_d < D)


# 3) RotPE: rotate half-dim for [B, S, D] (here we will apply to Q and K)
#    We implement per-token, per-head rotation using cos/sin, assuming D=128 (split 64+64).
@triton.jit
def triton_rotate_pe(
    X_ptr,        # *fp32, [M, D], where M = B * num_attention_heads
    Cos_ptr,      # *fp32, [D] cos table
    Sin_ptr,      # *fp32, [D] sin table
    Y_ptr,        # *fp32, [M, D]
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_c,     # cos stride (1)
    stride_s,     # sin stride (1)
    stride_ym, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        half = D // 2
        q1 = x[:half]
        q2 = x[half:]
        c = tl.load(Cos_ptr + offs_d * stride_c, mask=offs_d < D, other=1.0)
        s = tl.load(Sin_ptr + offs_d * stride_s, mask=offs_d < D, other=0.0)
        q1f = q1 * c[:half] + (-q2) * s[:half]
        q2f = q2 * c[half:] + q1 * s[half:]
        y = tl.zeros((D,), dtype=tl.float32)
        y[:half] = q1f
        y[half:] = q2f
        tl.store(Y_ptr + pid * stride_ym + offs_d * stride_yd, y, mask=offs_d < D)


# 4) Grouped Query Attention expansion: expand num_key_value_heads -> num_attention_heads * groups
#    We implement K (and similarly V). We write repeated rows into an expanded [M_exp, D] where
#    M_exp = B * num_attention_heads, and we repeat each of B * num_key_value_heads rows num_key_value_groups times.
@triton.jit
def triton_grouped_q_repeat(
    K_ptr,        # *fp32, [B*KH, D] (original K)
    ExpK_ptr,     # *fp32, [B*NAH, D] (expanded K, NAH*num_key_value_groups rows)
    B: tl.constexpr, KH: tl.constexpr, NAH: tl.constexpr, GH: tl.constexpr, D: tl.constexpr,
    stride_kb, stride_kd,
    stride_eb, stride_ed,
    BLOCK_D: tl.constexpr,
):
    # grid: (B * NAH, 1)
    pid = tl.program_id(0)
    b = pid // NAH
    kh = pid % NAH
    orig_index = b * KH + kh
    for g in range(NAH // KH * GH):
        exp_index = b * (NAH * GH) + (kh * GH + g)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            k = tl.load(K_ptr + orig_index * stride_kb + offs_d * stride_kd, mask=offs_d < D, other=0.0)
            tl.store(ExpK_ptr + exp_index * stride_eb + offs_d * stride_ed, k, mask=offs_d < D)


# 5) Score matmul: S[M, S] = Q[M, D] @ K^T[S, D] (per (batch, head) row)
@triton.jit
def triton_score_matmul(
    Q_ptr,        # *fp32, [M, D]
    Kt_ptr,       # *fp32, [S, D] (K transposed: [S, D])
    S_ptr,        # *fp32, [M, S]
    M: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qd,
    stride_st, stride_sd,   # Kt strides
    stride_sm, stride_ss,   # S strides
    scaling,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(Q_ptr + pid_m * stride_qm + offs_d * stride_qd, mask=offs_d < D, other=0.0)  # [BD]
        kt = tl.load(Kt_ptr + offs_s[:, None] * stride_st + offs_d[None, :] * stride_sd,
                     mask=(offs_s[:, None] < S) & (offs_d[None, :] < D),
                     other=0.0)  # [BS, BD]
        acc += tl.sum(kt * q[None, :], axis=1)  # [BS]
    acc = acc * scaling
    tl.store(S_ptr + pid_m * stride_sm + offs_s * stride_ss, acc, mask=offs_s < S)


# 6) Row-wise softmax over sequence dimension (normalize S[M, S] per row)
@triton.jit
def triton_softmax_rows(
    S_ptr,        # *fp32, [M, S]
    Soft_ptr,     # *fp32, [M, S]
    M: tl.constexpr, S: tl.constexpr,
    stride_sm, stride_ss,
    stride_sm_out, stride_ss_out,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    # compute max
    m = -1e30
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        x = tl.load(S_ptr + pid * stride_sm + offs_s * stride_ss, mask=offs_s < S, other=-1e30)
        m = tl.maximum(m, tl.max(x, axis=0))
    # compute exp and sum
    sum_exp = 0.0
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        x = tl.load(S_ptr + pid * stride_sm + offs_s * stride_ss, mask=offs_s < S, other=-1e30)
        e = tl.exp(x - m)
        tl.store(Soft_ptr + pid * stride_sm_out + offs_s * stride_ss_out, e, mask=offs_s < S)
        sum_exp += tl.sum(e, axis=0)
    # normalize
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        e = tl.load(Soft_ptr + pid * stride_sm_out + offs_s * stride_ss_out, mask=offs_s < S, other=0.0)
        y = e / sum_exp
        tl.store(Soft_ptr + pid * stride_sm_out + offs_s * stride_ss_out, y, mask=offs_s < S)


# 7) Final output: O[M, D] = Soft[M, S] @ V[M, D]
@triton.jit
def triton_final_output(
    Soft_ptr,     # *fp32, [M, S]
    V_ptr,        # *fp32, [M, D]
    O_ptr,        # *fp32, [M, D]
    M: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sm, stride_ss,
    stride_vm, stride_vd,
    stride_om, stride_od,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        soft = tl.load(Soft_ptr + pid * stride_sm + offs_s * stride_ss, mask=offs_s < S, other=0.0)  # [BS]
        v = tl.load(V_ptr + pid * stride_vm + offs_d * stride_vd, mask=offs_d < D, other=0.0)        # [BD]
        acc += tl.sum(soft[:, None] * v[None, :], axis=0)  # [BD]
    tl.store(O_ptr + pid * stride_om + offs_d * stride_od, acc, mask=offs_d < D)


# =========================
# ModelNew: Triton-only forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim=128, num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        # Precompute half
        self.half_dim = hidden_dim // 2

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        """
        Triton-only implementation of the given forward:
        - Dense no-bias GEMM for Q, K, V
        - RMSNorm for Q and K
        - RotPE for Q and K
        - Grouped Query Attention expand K/V to 96 heads
        - Compute attention scores Q @ K^T, softmax over sequence
        - Final output softmax @ V, then output projection
        """
        device = hidden_states.device
        B, S, H = hidden_states.shape
        assert H == self.hidden_dim, "hidden_dim mismatch"
        assert self.num_attention_heads * self.num_key_value_groups == self.num_key_value_heads * self.num_key_value_groups, "Configuration mismatch"
        # Ensure dtypes
        dtype = torch.float32
        hidden = hidden_states.to(dtype).contiguous()

        # 0) Initialize outputs
        output = None

        # 1) Dense no-bias GEMM: Q = hidden @ q_proj_weight^T -> [B*S, H]
        #    We pass q_proj_weight^T [H, H] to kernel, producing Q [B*S, H]
        Q = torch.empty((B * self.num_attention_heads, self.hidden_dim), device=device, dtype=dtype)
        # A: hidden reshaped to [B*SAH, H], but here SAH=num_attention_heads? The original code sets hidden_states [B, S, H] and uses linear(F) then reshape. To simplify, we compute Q via a custom kernel using hidden as A and q_proj_weight^T as B. We need to load the weight; q_proj_weight is [H, H].
        q_w_t = q_proj_weight.t().to(dtype).contiguous()  # [H, H]
        grid_q = (triton.cdiv(B * self.num_attention_heads, 64), triton.cdiv(self.hidden_dim, 64))
        triton_batched_gemm_no_bias[grid_q](
            hidden.view(B * self.num_attention_heads, H), q_w_t,
            Q,
            B * self.num_attention_heads, H, H,
            hidden.view(B * self.num_attention_heads, H).stride(0), hidden.view(B * self.num_attention_heads, H).stride(1),
            q_w_t.stride(0), q_w_t.stride(1),
            Q.stride(0), Q.stride(1),
            64, 64, 64
        )

        # 2) RMSNorm for Q
        Q_norm = torch.empty_like(Q)
        triton_rmsnorm[(B * self.num_attention_heads,)](
            Q, q_norm_weight.to(dtype).contiguous(), Q_norm,
            B * self.num_attention_heads, self.hidden_dim,
            Q.stride(0), Q.stride(1),
            q_norm_weight.to(dtype).contiguous().stride(0),
            Q_norm.stride(0), Q_norm.stride(1),
            rms_norm_eps,
            128
        )

        # 3) RotPE for Q
        Q_rot = torch.empty_like(Q_norm)
        # Prepare cos/sin (cos is [D], sin is [D])
        cos_ = cos.to(dtype).contiguous()
        sin_ = sin.to(dtype).contiguous()
        triton_rotate_pe[(B * self.num_attention_heads,)](
            Q_norm, cos_, sin_, Q_rot,
            B * self.num_attention_heads, self.hidden_dim,
            Q_norm.stride(0), Q_norm.stride(1),
            cos_.stride(0), sin_.stride(0),
            Q_rot.stride(0), Q_rot.stride(1),
            128
        )

        # 4) Dense no-bias GEMM: K = hidden @ k_proj_weight^T -> [B*num_key_value_heads, H]
        K = torch.empty((B * self.num_key_value_heads, self.hidden_dim), device=device, dtype=dtype)
        k_w_t = k_proj_weight.t().to(dtype).contiguous()  # [H, H]
        grid_k = (triton.cdiv(B * self.num_key_value_heads, 64), triton.cdiv(self.hidden_dim, 64))
        triton_batched_gemm_no_bias[grid_k](
            hidden.view(B * self.num_key_value_heads, H), k_w_t,
            K,
            B * self.num_key_value_heads, self.hidden_dim, H,
            hidden.view(B * self.num_key_value_heads, H).stride(0), hidden.view(B * self.num_key_value_heads, H).stride(1),
            k_w_t.stride(0), k_w_t.stride(1),
            K.stride(0), K.stride(1),
            64, 64, 64
        )

        # 5) RMSNorm for K
        K_norm = torch.empty_like(K)
        triton_rmsnorm[(B * self.num_key_value_heads,)](
            K, k_norm_weight.to(dtype).contiguous(), K_norm,
            B * self.num_key_value_heads, self.hidden_dim,
            K.stride(0), K.stride(1),
            k_norm_weight.to(dtype).contiguous().stride(0),
            K_norm.stride(0), K_norm.stride(1),
            rms_norm_eps,
            128
        )

        # 6) RotPE for K
        K_rot = torch.empty_like(K_norm)
        triton_rotate_pe[(B * self.num_key_value_heads,)](
            K_norm, cos_, sin_, K_rot,
            B * self.num_key_value_heads, self.hidden_dim,
            K_norm.stride(0), K_norm.stride(1),
            cos_.stride(0), sin_.stride(0),
            K_rot.stride(0), K_rot.stride(1),
            128
        )

        # 7) Grouped Query Attention: expand K to [B*num_attention_heads, H]
        ExpK = torch.empty((B * self.num_attention_heads, self.hidden_dim), device=device, dtype=dtype)
        # Original K shape is [B*num_key_value_heads, H], expand by num_key_value_groups (12)
        grid_gqa = (B * self.num_attention_heads, 1)
        triton_grouped_q_repeat[grid_gqa](
            K_rot, ExpK,
            B, self.num_key_value_heads, self.num_attention_heads, self.num_key_value_groups, self.hidden_dim,
            K_rot.stride(0), K_rot.stride(1),
            ExpK.stride(0), ExpK.stride(1),
            128
        )

        # 8) Score matmul: S[M, S] where M = B*num_attention_heads, N=S, D=H
        S_out = torch.empty((B * self.num_attention_heads, S), device=device, dtype=dtype)
        # We need Kt: [S, H] -> use K_rot.transpose(0,1)
        Kt = K_rot.transpose(0, 1).contiguous()  # [S, H]
        grid_score = (B * self.num_attention_heads, 1)
        triton_score_matmul[grid_score](
            Q_rot, Kt, S_out,
            B * self.num_attention_heads, S, self.hidden_dim,
            Q_rot.stride(0), Q_rot.stride(1),
            Kt.stride(0), Kt.stride(1),
            S_out.stride(0), S_out.stride(1),
            1.0 / (self.hidden_dim ** 0.5),
            128, 64
        )

        # 9) Row-wise softmax over sequence dimension (S_out is logits scaled by 1/sqrt(H))
        Soft = torch.empty_like(S_out)
        grid_softmax = (B * self.num_attention_heads, 1)
        triton_softmax_rows[grid_softmax](
            S_out, Soft,
            B * self.num_attention_heads, S,
            S_out.stride(0), S_out.stride(1),
            Soft.stride(0), Soft.stride(1),
            128
        )

        # 10) Final output: O[M, H] = Soft[M, S] @ V[M, H]
        # Compute V using hidden @ v_proj_weight^T -> [B*SAH, H]
        V = torch.empty((B * self.num_attention_heads, self.hidden_dim), device=device, dtype=dtype)
        v_w_t = v_proj_weight.t().to(dtype).contiguous()  # [H, H]
        grid_v = (triton.cdiv(B * self.num_attention_heads, 64), triton.cdiv(self.hidden_dim, 64))
        triton_batched_gemm_no_bias[grid_v](
            hidden.view(B * self.num_attention_heads, H), v_w_t,
            V,
            B * self.num_attention_heads, self.hidden_dim, H,
            hidden.view(B * self.num_attention_heads, H).stride(0), hidden.view(B * self.num_attention_heads, H).stride(1),
            v_w_t.stride(0), v_w_t.stride(1),
            V.stride(0), V.stride(1),
            64, 64, 64
        )
        O = torch.empty((B * self.num_attention_heads, self.hidden_dim), device=device, dtype=dtype)
        triton_final_output[(B * self.num_attention_heads,)](
            Soft, V, O,
            B * self.num_attention_heads, S, self.hidden_dim,
            Soft.stride(0), Soft.stride(1),
            V.stride(0), V.stride(1),
            O.stride(0), O.stride(1),
            128, 64
        )

        # 11) Transpose and reshape to [B, S, num_attention_heads*hidden_dim] -> [B, S, 12288]
        attn_out = O.view(B, S, self.num_attention_heads * self.hidden_dim)

        # 12) Output projection: attn_out @ o_proj_weight^T (no bias)
        M_proj = B * S
        K_proj = self.num_attention_heads * self.hidden_dim
        N_proj = self.hidden_dim  # output hidden dim
        A_proj = attn_out.view(M_proj, K_proj).contiguous()
        Bt_proj = o_proj_weight.t().to(dtype).contiguous()  # [hidden_dim, num_attention_heads*hidden_dim]
        output = torch.empty((M_proj, N_proj), device=device, dtype=dtype)
        grid_out_proj = (triton.cdiv(M_proj, 64), triton.cdiv(N_proj, 64))
        triton_batched_gemm_no_bias[grid_out_proj](
            A_proj, Bt_proj, output,
            M_proj, N_proj, K_proj,
            A_proj.stride(0), A_proj.stride(1),
            Bt_proj.stride(0), Bt_proj.stride(1),
            output.stride(0), output.stride(1),
            64, 64, 64
        )

        return output.view(B, S, self.hidden_dim)


def run(*args):
    return ModelNew()(*args)
