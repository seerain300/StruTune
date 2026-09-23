import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

# 1) Batched GEMM (no bias): C[M, N] = A[M, K] @ B[K, N], where B is weight^T.
# A is [M, K], B is [K, N], C is [M, N].
@triton.jit
def triton_batched_gemm_no_bias(
    A_ptr,        # *fp32, [M, K]
    B_ptr,        # *fp32, [K, N]
    C_ptr,        # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# 2) RMSNorm per row (M rows, D dim): y[i, :] = x[i, :] * weight / sqrt(mean(x[i]^2) + eps)
@triton.jit
def triton_rmsnorm(
    X_ptr,        # *fp32, [M, D]
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
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(sumsq / D + eps)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        w = tl.load(Weight_ptr + offs_d * stride_w, mask=offs_d < D, other=1.0)
        y = x * inv * w
        tl.store(Y_ptr + pid * stride_ym + offs_d * stride_yd, y, mask=offs_d < D)


# 3) RotPE: rotate half-dim for [M, D], D assumed 128. For each row: q1 = x[:D//2], q2 = x[D//2:]; y = q1*cos - q2*sin for first half and q2*cos + q1*sin for second half.
@triton.jit
def triton_rotate_pe(
    X_ptr,        # *fp32, [M, D]
    Cos_ptr,      # *fp32, [D]
    Sin_ptr,      # *fp32, [D]
    Y_ptr,        # *fp32, [M, D]
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_c, stride_s,
    stride_ym, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + pid * stride_xm + offs_d * stride_xd, mask=offs_d < D, other=0.0)
        half = D // 2
        q1 = x[:half]
        q2 = x[half:]
        c = tl.load(Cos_ptr + offs_d * stride_c, mask=offs_d < D, other=1.0)
        s = tl.load(Sin_ptr + offs_d * stride_s, mask=offs_d < D, other=0.0)
        # for first half: y1 = q1*c - q2*s
        # for second half: y2 = q2*c + q1*s
        y1 = q1 * c[:half] - q2 * s[:half]
        y2 = q2 * c[half:] + q1 * s[half:]
        y = tl.zeros((D,), dtype=tl.float32)
        y[:half] = y1
        y[half:] = y2
        tl.store(Y_ptr + pid * stride_ym + offs_d * stride_yd, y, mask=offs_d < D)


# 4) Grouped Query Attention expansion: expand K/V from [B, S, num_key_value_heads, D] to [B, S, num_attention_heads, D] by repeating each key/value head across groups.
# This kernel takes K_raw [B, S, num_key_value_heads, D], expands to K_exp [B, S, num_attention_heads, D] by repeating num_key_value_groups times for each original head.
@triton.jit
def triton_gqa_expand(
    K_raw_ptr,      # *fp32, [B, S, num_key_value_heads, D]
    K_exp_ptr,      # *fp32, [B, S, num_attention_heads, D]
    B: tl.constexpr, S: tl.constexpr, Nk: tl.constexpr, Na: tl.constexpr, D: tl.constexpr,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_keb, stride_kes, stride_keh, stride_ked,
    GROUPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h_out = tl.program_id(2)  # output head index in [0, Na)
    pid_rep = tl.program_id(3)    # repetition index within groups

    orig_h = pid_h_out % Nk
    rep = pid_rep
    if rep >= GROUPS:
        return

    # copy K_raw[pid_b, pid_s, orig_h, :] to K_exp[pid_b, pid_s, pid_h_out, :]
    src = tl.load(
        K_raw_ptr + pid_b * stride_kb + pid_s * stride_ks + orig_h * stride_kh + tl.arange(0, D) * stride_kd
    )
    tl.store(
        K_exp_ptr + pid_b * stride_keb + pid_s * stride_kes + pid_h_out * stride_keh + tl.arange(0, D) * stride_ked,
        src
    )


# 5) Attention scores: S[M, N] = Q[M, D] @ K^T[N, D], where M = B * num_attention_heads, N = S
@triton.jit
def triton_score_matmul(
    Q_ptr,         # *fp32, [M, D]
    Kt_ptr,        # *fp32, [N, D] (K^T)
    S_ptr,         # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qd,
    stride_kn, stride_kd,  # K^T strides
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row index over M
    pid_n = tl.program_id(1)  # col index over N
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
        kt = tl.load(
            Kt_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0
        )  # [BLOCK_N, BLOCK_D]
        acc += tl.dot(q, kt)  # [BLOCK_M, BLOCK_N]

    tl.store(
        S_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# 6) Row-wise softmax over N dimension (sequence), S[M, N]
@triton.jit
def triton_softmax_row(
    S_ptr,         # *fp32, [M, N]
    Out_ptr,       # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr,
    stride_sm, stride_sn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row index
    # compute max across N
    max_val = -float('inf')
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        s = tl.load(S_ptr + pid_m * stride_sm + offs_n * stride_sn, mask=offs_n < N, other=-float('inf'))
        block_max = tl.max(s, axis=0)
        max_val = tl.maximum(max_val, block_max)
    # compute sum of exp(s - max)
    sum_exp = 0.0
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        s = tl.load(S_ptr + pid_m * stride_sm + offs_n * stride_sn, mask=offs_n < N, other=-float('inf'))
        e = tl.exp(s - max_val)
        sum_exp += tl.sum(e, axis=0)
    # write normalized
    inv_sum = 1.0 / sum_exp
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        s = tl.load(S_ptr + pid_m * stride_sm + offs_n * stride_sn, mask=offs_n < N, other=-float('inf'))
        e = tl.exp(s - max_val) * inv_sum
        tl.store(Out_ptr + pid_m * stride_om + offs_n * stride_on, e, mask=offs_n < N)


# 7) Final output: softmax_scores [M, N] @ V[M, N, D] -> out[M, D], where M = B * num_attention_heads, N = seq_len, D = head_dim
@triton.jit
def triton_final_output_row(
    S_ptr,         # *fp32, [M, N] softmax scores
    V_ptr,         # *fp32, [M, N, D] value states (M = B * Na, N = seq_len, D = head_dim)
    Out_ptr,       # *fp32, [M, D] result
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_sm, stride_sn,
    stride_vm, stride_vn, stride_vd,
    stride_om, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            s = tl.load(S_ptr + pid_m * stride_sm + offs_n * stride_sn, mask=offs_n < N, other=0.0)  # [BLOCK_N]
            v = tl.load(
                V_ptr + pid_m * stride_vm + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
                other=0.0
            )  # [BLOCK_N, BLOCK_D]
            acc += tl.sum(s[:, None] * v, axis=0)  # reduce over N
        tl.store(Out_ptr + pid_m * stride_om + offs_d * stride_od, acc, mask=offs_d < D)


# =========================
# ModelNew: Triton forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim  # head_dim
        # constants as per original: num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float,
                ) -> torch.Tensor:
        # hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        assert H == self.hidden_dim, "hidden_dim mismatch"
        device = hidden_states.device
        dtype = torch.float32

        # 1) Compute Q_raw, K_raw, V_raw via dense GEMM (no bias) using hidden_states [B*S, H] and weight^T [H, H]
        # Prepare A as [B*S, H]
        A_q = hidden_states.view(B * S, H).contiguous().to(dtype)
        # weight q_proj_weight [H, H] -> Bq [H, H]
        Bq = q_proj_weight.to(dtype).contiguous()
        C_q = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_batched_gemm_no_bias[(B * S, cdiv(H, 64))](  # grid: (M, cdiv(N, BLOCK))
            A_q, Bq, C_q,
            B * S, H, H,
            A_q.stride(0), A_q.stride(1),
            Bq.stride(0), Bq.stride(1),
            C_q.stride(0), C_q.stride(1),
            64, 64, 64
        )
        Q_raw = C_q.view(B, S, H)

        # K_raw and V_raw similarly
        A_k = hidden_states.view(B * S, H).contiguous().to(dtype)
        Bk = k_proj_weight.to(dtype).contiguous()
        C_k = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_batched_gemm_no_bias[(B * S, cdiv(H, 64))](  # grid: (M, cdiv(N, BLOCK))
            A_k, Bk, C_k,
            B * S, H, H,
            A_k.stride(0), A_k.stride(1),
            Bk.stride(0), Bk.stride(1),
            C_k.stride(0), C_k.stride(1),
            64, 64, 64
        )
        K_raw = C_k.view(B, S, H)

        A_v = hidden_states.view(B * S, H).contiguous().to(dtype)
        Bv = v_proj_weight.to(dtype).contiguous()
        C_v = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_batched_gemm_no_bias[(B * S, cdiv(H, 64))](  # grid: (M, cdiv(N, BLOCK))
            A_v, Bv, C_v,
            B * S, H, H,
            A_v.stride(0), A_v.stride(1),
            Bv.stride(0), Bv.stride(1),
            C_v.stride(0), C_v.stride(1),
            64, 64, 64
        )
        V_raw = C_v.view(B, S, H)

        # 2) RMSNorm for query and key
        # For RMSNorm, we need per (batch, seq, head). Since we have [B, S, H], we treat M=B*S and each row corresponds to a token position across heads.
        # But RMSNorm per token per head is over H dim. We can apply per token across H:
        # Prepare [B*S, H] tensors
        Q_raw_flat = Q_raw.reshape(B * S, H).contiguous()
        K_raw_flat = K_raw.reshape(B * S, H).contiguous()
        # Normalize Q
        Q_norm = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_rmsnorm[(B * S,)](Q_raw_flat, q_norm_weight.to(dtype), Q_norm, B * S, H, Q_raw_flat.stride(0), Q_raw_flat.stride(1), q_norm_weight.to(dtype).stride(0), Q_norm.stride(0), Q_norm.stride(1), rms_norm_eps, 128)
        # Normalize K
        K_norm = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_rmsnorm[(B * S,)](K_raw_flat, k_norm_weight.to(dtype), K_norm, B * S, H, K_raw_flat.stride(0), K_raw_flat.stride(1), k_norm_weight.to(dtype).stride(0), K_norm.stride(0), K_norm.stride(1), rms_norm_eps, 128)

        # 3) RotPE for Q and K
        # cos, sin are [H] in original. Ensure float32.
        cos_dev = cos.to(dtype)
        sin_dev = sin.to(dtype)
        Q_rot = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_rotate_pe[(B * S,)](Q_norm, cos_dev, sin_dev, Q_rot, B * S, H, Q_norm.stride(0), Q_norm.stride(1), cos_dev.stride(0), sin_dev.stride(0), Q_rot.stride(0), Q_rot.stride(1), 128)
        K_rot = torch.empty((B * S, H), device=device, dtype=torch.float32)
        triton_rotate_pe[(B * S,)](K_norm, cos_dev, sin_dev, K_rot, B * S, H, K_norm.stride(0), K_norm.stride(1), cos_dev.stride(0), sin_dev.stride(0), K_rot.stride(0), K_rot.stride(1), 128)

        # 4) Grouped Query Attention: expand K_rot and V_raw to 96 heads
        K_rot_exp = torch.empty((B, S, self.num_attention_heads, H), device=device, dtype=torch.float32)
        V_raw_exp = torch.empty((B, S, self.num_attention_heads, H), device=device, dtype=torch.float32)
        # launch grid over (B, S, num_key_value_heads * num_key_value_groups)
        grid_gqa = (B, S, self.num_key_value_heads, self.num_key_value_groups)
        triton_gqa_expand[grid_gqa](
            K_rot.reshape(B, S, self.num_key_value_heads, H),
            K_rot_exp,
            B, S, self.num_key_value_heads, self.num_attention_heads, H,
            K_rot.reshape(B, S, self.num_key_value_heads, H).stride(0), K_rot.reshape(B, S, self.num_key_value_heads, H).stride(1), K_rot.reshape(B, S, self.num_key_value_heads, H).stride(2), K_rot.reshape(B, S, self.num_key_value_heads, H).stride(3),
            K_rot_exp.stride(0), K_rot_exp.stride(1), K_rot_exp.stride(2), K_rot_exp.stride(3),
            self.num_key_value_groups,
        )
        # same for V_raw
        V_raw_exp = torch.empty((B, S, self.num_attention_heads, H), device=device, dtype=torch.float32)
        triton_gqa_expand[grid_gqa](
            V_raw.reshape(B, S, self.num_key_value_heads, H),
            V_raw_exp,
            B, S, self.num_key_value_heads, self.num_attention_heads, H,
            V_raw.reshape(B, S, self.num_key_value_heads, H).stride(0), V_raw.reshape(B, S, self.num_key_value_heads, H).stride(1), V_raw.reshape(B, S, self.num_key_value_heads, H).stride(2), V_raw.reshape(B, S, self.num_key_value_heads, H).stride(3),
            V_raw_exp.stride(0), V_raw_exp.stride(1), V_raw_exp.stride(2), V_raw_exp.stride(3),
            self.num_key_value_groups,
        )

        # 5) Compute attention scores S[M, S] where M = B * num_attention_heads
        # We use Q_rot [B*S, H] and K_rot_exp [B, S, Na, H]; need K^T [Na, S, H] per (b,h). For simplicity, flatten to [M, H].
        Q_rot_flat = Q_rot.reshape(B * S, H).contiguous()
        # Build Kt for each (b, head), then call score_matmul. We can compute M = B*Na, N = S.
        # But we need K^T per head. Let’s build Kt as [M, S, H] where each row is K_rot_exp[b,s,head,:].
        # To use triton_score_matmul, we need Kt of shape [N, D] per row of M. We will loop over M, but Triton expects static shapes; better to compute Kt as [M, H] by indexing.
        # Instead, we construct Kt rows by combining per (b, h). Since we already have K_rot_exp [B,S,Na,H], we can flatten and then index: Kt[M, H].
        # For generality, we can rebuild Kt per (b,h) as vector per M. Simpler: build Kt[M,H] by selecting K_rot_exp[b,s,h,:]. We will generate a [M,H] tensor by indexing K_rot_exp.

        # Reconstruct Kt[M,H] from K_rot_exp
        Kt_all = torch.empty((B * self.num_attention_heads, H), device=device, dtype=torch.float32)
        for m in range(B * self.num_attention_heads):
            # determine b and h
            b = m // self.num_attention_heads
            h = m % self.num_attention_heads
            # iterate s and accumulate? Not needed: we can just gather K_rot_exp[b,s,h,:] into a vector by taking all s at that (b,h). But we need K_rot_exp[b,s,h,:] for all s. Simpler: build Kt by flattening K_rot_exp.
            # K_rot_exp has shape [B, S, Na, H]; flatten over s to build rows. Each row corresponds to a (b, h) and H dimension.
            # To do this, we need a loop over s. Triton kernels require static shapes; we’ll do this in PyTorch for simplicity: torch.cat over s dimension.
            # However, to keep Triton-only, we’ll implement a kernel to fill Kt by copying per (b,h) across s.
            # Define a kernel: copy K_rot_exp[b, :, h, :] into rows Kt_all[m, :]
            # We need to iterate s in the kernel? Triton allows loops with runtime bounds. We’ll pass B,S,Na and do it.
            # Better: precompute Kt_all by concatenating across s for each (b,h). We’ll do that in Python loop:
            # But since we must use Triton, we implement a tiny kernel that copies K_rot_exp[b, s, h, :] to Kt_all[m, :] per iteration. This is fine: Triton supports loops with runtime values.

        # Implement a kernel to fill Kt_all[M,H] from K_rot_exp[B,S,Na,H]:
        Kt_all = torch.empty((B * self.num_attention_heads, H), device=device, dtype=torch.float32)
        # Launch grid over M
        triton_fill_Kt[(B * self.num_attention_heads,)](
            K_rot_exp, Kt_all,
            B, S, self.num_attention_heads, H,
            K_rot_exp.stride(0), K_rot_exp.stride(1), K_rot_exp.stride(2), K_rot_exp.stride(3),
            Kt_all.stride(0), Kt_all.stride(1),
            128
        )

        # Now compute S[M, S] = Q_rot_flat[M,H] @ Kt_all^T
        S_out = torch.empty((B * self.num_attention_heads, S), device=device, dtype=torch.float32)
        triton_score_matmul[(B * self.num_attention_heads, cdiv(S, 64))](  # grid: (M, cdiv(N, BLOCK))
            Q_rot_flat, Kt_all, S_out,
            B * self.num_attention_heads, S, H,
            Q_rot_flat.stride(0), Q_rot_flat.stride(1),
            Kt_all.stride(0), Kt_all.stride(1),  # Kt_all[M,H] strides
            S_out.stride(0), S_out.stride(1),
            64, 128, 64
        )
        # Scale by 1/sqrt(H)
        scaling = 1.0 / (H ** 0.5)
        S_out = S_out * scaling

        # 6) Softmax over sequence dimension per row (M = B*Na)
        Out_soft = torch.empty_like(S_out)
        triton_softmax_row[(B * self.num_attention_heads,)](
            S_out, Out_soft,
            B * self.num_attention_heads, S,
            S_out.stride(0), S_out.stride(1),
            Out_soft.stride(0), Out_soft.stride(1),
            128
        )

        # 7) Final attention output: Out_soft[M,S] @ V_raw_exp[M,S,H] -> Out[M,H]
        Out_final = torch.empty((B * self.num_attention_heads, H), device=device, dtype=torch.float32)
        triton_final_output_row[(B * self.num_attention_heads,)](
            Out_soft, V_raw_exp.reshape(B * self.num_attention_heads, S, H),
            Out_final,
            B * self.num_attention_heads, S, H,
            Out_soft.stride(0), Out_soft.stride(1),
            V_raw_exp.reshape(B * self.num_attention_heads, S, H).stride(0), V_raw_exp.reshape(B * self.num_attention_heads, S, H).stride(1), V_raw_exp.reshape(B * self.num_attention_heads, S, H).stride(2),
            Out_final.stride(0), Out_final.stride(1),
            128, 128
        )

        # Reshape to [B, S, Na * H] and output projection
        attn_out = Out_final.view(B, S, self.num_attention_heads * self.hidden_dim)
        # Output projection (no bias): attn_out @ o_proj_weight^T
        A_proj = attn_out.view(B * S, self.num_attention_heads * self.hidden_dim).contiguous().to(torch.float32)
        Bt_proj = o_proj_weight.t().contiguous().to(torch.float32)  # [hidden_dim, num_attention_heads * hidden_dim]
        output = torch.empty((B * S, self.hidden_dim), device=device, dtype=torch.float32)
        triton_batched_gemm_no_bias[(B * S, cdiv(self.hidden_dim, 64))](  # grid: (M, cdiv(N, BLOCK))
            A_proj, Bt_proj, output,
            B * S, self.hidden_dim, self.num_attention_heads * self.hidden_dim,
            A_proj.stride(0), A_proj.stride(1),
            Bt_proj.stride(0), Bt_proj.stride(1),
            output.stride(0), output.stride(1),
            64, 64, 64
        )

        return output.view(B, S, self.hidden_dim)


def run(*args):
    return ModelNew()(*args)
