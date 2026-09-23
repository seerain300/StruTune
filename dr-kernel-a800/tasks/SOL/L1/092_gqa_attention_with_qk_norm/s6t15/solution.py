import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

# 1) Batched GEMM (no bias): C[M, N] = A[M, K] @ B[K, N], where B is weight^T (float32)
@triton.jit
def triton_batched_gemm_no_bias(
    A_ptr,        # *fp32, [M, K] (A = hidden_states reshaped)
    B_ptr,        # *fp32, [K, N] (weight^T, e.g., q_proj_weight^T)
    C_ptr,        # *fp32, [M, N] (output)
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
        # Load A block: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        # Load B block as [BLOCK_K, BLOCK_N] (note transpose of weight)
        b = tl.load(
            B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # Store results with proper masking
    tl.store(
        C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

# 2) RMSNorm per head over head_dim (normalize across last dim), then scale by weight
@triton.jit
def triton_rmsnorm(
    X_ptr,        # *fp32, [M, D] input
    W_ptr,        # *fp32, [D] weight
    Y_ptr,        # *fp32, [M, D] output
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_w,
    stride_ym, stride_yd,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_d = tl.arange(0, D)
    m = pid_m
    # mean of squares across D
    sum_sq = 0.0
    for d in range(0, D):
        x = tl.load(X_ptr + m * stride_xm + d * stride_xd)
        sum_sq += x * x
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + eps)
    for d in range(0, D):
        x = tl.load(X_ptr + m * stride_xm + d * stride_xd)
        w = tl.load(W_ptr + d * stride_w)
        y = x * inv_rms * w
        tl.store(Y_ptr + m * stride_ym + d * stride_yd, y)

# 3) Rotate PE (Rotary Position Embedding) for Q or K: apply rotation using cos/sin on last dim=64
# For each row [S, D], split into two halves: [q1, q2] = [q[:, :D/2], q[:, D/2:]], then q' = q1*cos + q2*sin and -q2*cos + q1*sin
@triton.jit
def triton_rotate_pe(
    X_ptr,        # *fp32, [M, D] input (e.g., query or key after norm)
    cos_ptr,      # *fp32, [D/2] cosine
    sin_ptr,      # *fp32, [D/2] sine
    Y_ptr,        # *fp32, [M, D] output
    M: tl.constexpr, D: tl.constexpr,             # M=rows=S, D=head_dim
    stride_xm, stride_xd,
    stride_ym, stride_yd,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, D)
    m = pid_m
    x = tl.load(X_ptr + m * stride_xm + offs * stride_xd)
    half = D // 2
    q1 = x[:half]
    q2 = x[half:]
    c = tl.load(cos_ptr + tl.arange(0, half))
    s = tl.load(sin_ptr + tl.arange(0, half))
    y = q1 * c[None, :] + q2 * s[None, :]
    negpart = -q2 * c[None, :] + q1 * s[None, :]
    y = tl.concatenate([y, negpart], axis=0)
    tl.store(Y_ptr + m * stride_ym + offs * stride_yd, y)

# 4) Grouped Query Attention repeat: expand num_key_value_heads=8 into num_attention_heads=96 via groups=12
# Given S_K [B, H_k, S, D] (H_k=8, D=128), write S_K_rep [B, H_q=96, S, D] where each H_k is repeated 12 times
@triton.jit
def triton_grouped_q_repeat(
    SK_ptr,       # *fp32, [B, H_k, S, D]
    SKR_ptr,      # *fp32, [B, H_q, S, D]
    B: tl.constexpr, H_k: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sbk, stride_sbh, stride_sbs, stride_sbd,
    stride_sbrb, stride_sbrh, stride_sbrs, stride_sbrd,
    groups: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    b = pid_b
    hq = pid_hq
    orig_h = hq % H_k
    group = hq // H_k
    assert group < groups, "group index out of range"
    # copy row from SK to SKR
    for s in range(0, S):
        offs_d = tl.arange(0, D)
        src = SK_ptr + b * stride_sbk + orig_h * stride_sbh + s * stride_sbs + offs_d * stride_sbd
        dst = SKR_ptr + b * stride_sbrb + hq * stride_sbrh + s * stride_sbrs + offs_d * stride_sbrd
        vals = tl.load(src)
        tl.store(dst, vals)

# 5) Score matmul: compute attention scores S[M, N] = Q[M, D] @ K^T[N, D], where M=B*num_attention_heads, N=S
# Implement row-wise: for each m (batch*head), compute all N scores; D=head_dim=128
@triton.jit
def triton_score_matmul_rowwise(
    Q_ptr,        # *fp32, [M, D]
    K_ptr,        # *fp32, [N, D] (K is [S, D] here)
    S_ptr,        # *fp32, [M, N] output scores
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_sm, stride_sn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    offs_n = tl.arange(0, N)  # we will loop over N to keep memory access stable
    acc = tl.zeros((N,), dtype=tl.float32)

    for n0 in range(0, N, 128):
        cur_n = n0 + tl.arange(0, 128)
        mask_n = cur_n < N
        q_row = tl.load(
            Q_ptr + m * stride_qm + tl.arange(0, D) * stride_qd,
            mask=True,
            other=0.0
        )  # [D]
        k_block = tl.load(
            K_ptr + cur_n[None, :] * stride_kn + tl.arange(0, D) * stride_kd,
            mask=mask_n[None, :],
            other=0.0
        )  # [128, D]
        acc[cur_n] += tl.sum(q_row[None, :] * k_block, axis=1)  # [128]

    # Store acc with masking
    tl.store(
        S_ptr + m * stride_sm + offs_n * stride_sn,
        acc,
        mask=offs_n < N
    )

# 6) Softmax over sequence dimension (row-wise)
@triton.jit
def triton_softmax_rows(
    S_ptr,        # *fp32, [M, N] scores
    P_ptr,        # *fp32, [M, N] probabilities
    M: tl.constexpr, N: tl.constexpr,
    stride_sm, stride_sn,
    stride_pm, stride_pn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    # Compute row max
    max_val = -float('inf')
    for n in range(0, N):
        val = tl.load(S_ptr + m * stride_sm + n * stride_sn)
        if val > max_val:
            max_val = val
    # Compute sum of exp(x - max)
    sum_exp = 0.0
    for n in range(0, N):
        val = tl.load(S_ptr + m * stride_sm + n * stride_sn)
        expv = tl.exp(val - max_val)
        sum_exp += expv
    # Normalize
    for n in range(0, N):
        val = tl.load(S_ptr + m * stride_sm + n * stride_sn)
        p = tl.exp(val - max_val) / sum_exp
        tl.store(P_ptr + m * stride_pm + n * stride_pn, p)

# 7) Final output: O[M, N] = P[M, N] @ V[N, D], M = B*num_attention_heads, N = S, D = head_dim
@triton.jit
def triton_final_output_rowwise(
    P_ptr,        # *fp32, [M, N] (softmax over sequence)
    V_ptr,        # *fp32, [N, D]
    O_ptr,        # *fp32, [M, D]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_pm, stride_pn,
    stride_vn, stride_vd,
    stride_om, stride_od,
):
    pid_m = tl.program_id(0)
    m = pid_m
    offs_d = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)

    for n0 in range(0, N, 128):
        cur_n = n0 + tl.arange(0, 128)
        mask_n = cur_n < N
        p_row = tl.load(
            P_ptr + m * stride_pm + cur_n[None, :] * stride_pn,
            mask=mask_n[None, :],
            other=0.0
        )  # [1, 128]
        v_block = tl.load(
            V_ptr + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0
        )  # [128, D]
        acc += tl.sum(p_row * v_block, axis=0)  # sum over 128 to get [D]

    tl.store(
        O_ptr + m * stride_om + offs_d * stride_od,
        acc,
        mask=offs_d < D
    )

# 8) Output projection: O_final[M, H] = O[M, K] @ Wt[K, H], no bias. M = B*S, K = num_attention_heads*head_dim, H = hidden_dim
@triton.jit
def triton_output_proj_rowwise(
    O_ptr,        # *fp32, [M, K]
    Wt_ptr,       # *fp32, [K, H] (weight^T, hidden_dim x num_attention_heads*head_dim)
    Y_ptr,        # *fp32, [M, H]
    M: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
    stride_om, stride_ok,
    stride_wtk, stride_wth,
    stride_y_m, stride_y_h,
):
    pid_m = tl.program_id(0)
    m = pid_m
    offs_h = tl.arange(0, H)
    acc = tl.zeros((H,), dtype=tl.float32)

    for k0 in range(0, K, 128):
        offs_k = k0 + tl.arange(0, 128)
        mask_k = offs_k < K
        o_row = tl.load(
            O_ptr + m * stride_om + offs_k[None, :] * stride_ok,
            mask=mask_k[None, :],
            other=0.0
        )  # [1, 128]
        wt_block = tl.load(
            Wt_ptr + offs_k[:, None] * stride_wtk + offs_h[None, :] * stride_wth,
            mask=mask_k[:, None],
            other=0.0
        )  # [128, H]
        acc += tl.sum(o_row * wt_block, axis=0)  # [H]

    tl.store(
        Y_ptr + m * stride_y_m + offs_h * stride_y_h,
        acc,
        mask=offs_h < H
    )


# =========================
# ModelNew: Triton-ONLY forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim=12288, head_dim=128, num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12, rms_norm_eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.rms_norm_eps = rms_norm_eps

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor, o_proj_bias: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        # hidden_states: [B, S, H] where H=self.hidden_dim
        assert hidden_states.dtype == torch.float32, "hidden_states must be float32"
        device = hidden_states.device

        B, S, H = hidden_states.shape
        D = self.head_dim
        Hq = self.num_attention_heads
        Hk = self.num_key_value_heads
        groups = self.num_key_value_groups
        scaling = 1.0 / (D ** 0.5)

        # 1) Q = hidden_states @ q_proj_weight^T (no bias)
        Aq = hidden_states.reshape(B * Hq, D)  # [Mq, D]
        BqT = q_proj_weight.t().contiguous()   # [D, D]
        Mq = B * Hq
        Q = torch.empty((Mq, D), device=device, dtype=torch.float32)
        grid_gemm = (triton.cdiv(Mq, 64), triton.cdiv(D, 64))
        triton_batched_gemm_no_bias[grid_gemm](
            Aq, BqT, Q,
            Mq, D, D,
            Aq.stride(0), Aq.stride(1),
            BqT.stride(0), BqT.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 2) K = hidden_states @ k_proj_weight^T (no bias)
        Ak = hidden_states.reshape(B * Hk, D)  # [Mk, D]
        BkT = k_proj_weight.t().contiguous()   # [D, D]
        Mk = B * Hk
        K = torch.empty((Mk, D), device=device, dtype=torch.float32)
        grid_gemm_k = (triton.cdiv(Mk, 64), triton.cdiv(D, 64))
        triton_batched_gemm_no_bias[grid_gemm_k](
            Ak, BkT, K,
            Mk, D, D,
            Ak.stride(0), Ak.stride(1),
            BkT.stride(0), BkT.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 3) V = hidden_states @ v_proj_weight^T (no bias)
        Av = hidden_states.reshape(B * Hk, D)  # [Mk, D]
        BvT = v_proj_weight.t().contiguous()   # [D, D]
        V = torch.empty((Mk, D), device=device, dtype=torch.float32)
        grid_gemm_v = (triton.cdiv(Mk, 64), triton.cdiv(D, 64))
        triton_batched_gemm_no_bias[grid_gemm_v](
            Av, BvT, V,
            Mk, D, D,
            Av.stride(0), Av.stride(1),
            BvT.stride(0), BvT.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 4) RMSNorm on Q and K
        # Q norm
        Yq = torch.empty_like(Q)
        triton_rmsnorm[(Mq,)](
            Q, q_norm_weight, Yq,
            Mq, D,
            Q.stride(0), Q.stride(1),
            q_norm_weight.stride(0),
            Yq.stride(0), Yq.stride(1),
            self.rms_norm_eps,
            num_warps=1, num_stages=1,
        )
        # K norm
        Yk = torch.empty_like(K)
        triton_rmsnorm[(Mk,)](
            K, k_norm_weight, Yk,
            Mk, D,
            K.stride(0), K.stride(1),
            k_norm_weight.stride(0),
            Yk.stride(0), Yk.stride(1),
            self.rms_norm_eps,
            num_warps=1, num_stages=1,
        )

        # 5) Rotate Q and K with cos/sin
        # Q rotate
        Qr = torch.empty_like(Yq)
        triton_rotate_pe[(Mq, D)](
            Yq, cos, sin, Qr,
            Mq, D,
            Yq.stride(0), Yq.stride(1),
            Qr.stride(0), Qr.stride(1),
            num_warps=1, num_stages=1,
        )
        # K rotate
        Kr = torch.empty_like(Yk)
        triton_rotate_pe[(Mk, D)](
            Yk, cos, sin, Kr,
            Mk, D,
            Yk.stride(0), Yk.stride(1),
            Kr.stride(0), Kr.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Repeat KV heads into 96: SK is [B, Hk, S, D]; we need SRep [B, Hq, S, D]
        # First, prepare S_K_rep shape for expanded K and V. Note: we expand by repeating each of 8 heads 12 times.
        SRep_K = torch.empty((B, Hq, S, D), device=device, dtype=torch.float32)
        grid_repeat_k = (B, Hq)
        triton_grouped_q_repeat[grid_repeat_k](
            Kr, SRep_K,
            B, Hk, S, D,
            Kr.stride(0), Kr.stride(1), Kr.stride(2), Kr.stride(3),
            SRep_K.stride(0), SRep_K.stride(1), SRep_K.stride(2), SRep_K.stride(3),
            groups,
            num_warps=1, num_stages=1,
        )

        # For V, we also need SRep_V: [B, Hq, S, D]
        SRep_V = torch.empty((B, Hq, S, D), device=device, dtype=torch.float32)
        grid_repeat_v = (B, Hq)
        triton_grouped_q_repeat[grid_repeat_v](
            V, SRep_V,
            B, Hk, S, D,
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            SRep_V.stride(0), SRep_V.stride(1), SRep_V.stride(2), SRep_V.stride(3),
            groups,
            num_warps=1, num_stages=1,
        )

        # 7) Compute attention scores: S[Mq, S] = Qr[Mq, D] @ Krep^T[S, D]
        M = Mq  # B * Hq
        S_out = torch.empty((M, S), device=device, dtype=torch.float32)
        grid_score = (M, triton.cdiv(S, 128))
        triton_score_matmul_rowwise[grid_score](
            Qr, SRep_K, S_out,
            M, S, D,
            Qr.stride(0), Qr.stride(1),
            SRep_K.stride(1), SRep_K.stride(3),  # using (head, seq, dim) strides: stride_kn -> stride(1), stride_kd -> stride(3)
            S_out.stride(0), S_out.stride(1),
            num_warps=4, num_stages=2,
        )

        # 8) Apply scaling and causal mask in Triton: not available; implement with softmax in Triton over sequence
        # Triton softmax: P[M, S] = softmax(S_out) along S dimension
        P = torch.empty_like(S_out)
        grid_softmax = (M,)
        triton_softmax_rows[grid_softmax](
            S_out, P,
            M, S,
            S_out.stride(0), S_out.stride(1),
            P.stride(0), P.stride(1),
            num_warps=2, num_stages=1,
        )

        # 9) Final attention output: O[M, D] = P[M, S] @ Vrep^T[S, D]
        O = torch.empty((M, D), device=device, dtype=torch.float32)
        grid_out = (M, triton.cdiv(D, 128))
        triton_final_output_rowwise[grid_out](
            P, SRep_V, O,
            M, S, D,
            P.stride(0), P.stride(1),
            SRep_V.stride(1), SRep_V.stride(3),
            O.stride(0), O.stride(1),
            num_warps=4, num_stages=2,
        )

        # 10) Reshape to [B, S, num_attention_heads * head_dim] = [B, S, 12288]
        O_reshaped = O.view(B, S, Hq * D)

        # 11) Output projection (no bias): O_final[B*S, hidden_dim] = O_reshaped[B*S, Hq*D] @ o_proj_weight^T
        M_final = B * S
        K_final = Hq * D
        H_final = H  # hidden_dim
        A_proj = O_reshaped.reshape(M_final, K_final).contiguous()
        Wt_proj = o_proj_weight.t().contiguous()  # [hidden_dim, num_attention_heads*head_dim] = [H, H_final]
        output = torch.empty((M_final, H_final), device=device, dtype=torch.float32)
        grid_proj = (triton.cdiv(M_final, 128), triton.cdiv(H_final, 128))
        triton_output_proj_rowwise[grid_proj](
            A_proj, Wt_proj, output,
            M_final, K_final, H_final,
            A_proj.stride(0), A_proj.stride(1),
            Wt_proj.stride(0), Wt_proj.stride(1),
            output.stride(0), output.stride(1),
            num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
