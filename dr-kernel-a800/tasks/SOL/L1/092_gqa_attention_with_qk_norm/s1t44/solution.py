import torch
import torch.nn as nn
import triton
import triton.language as tl


# Linear projection: Y[M, N] = X[M, K] @ W[N, K]^T + bias[N]
@triton.jit
def linear_fwd_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0
        )
        acc += tl.dot(x, w)

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    acc += bias[None, :]

    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# RMSNorm for each row along last dimension: y = x * rsqrt(mean(x^2) + eps)
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )
    sumsq = tl.sum(x * x, axis=1)  # [BLOCK_M]
    mean = sumsq / N
    inv_rms = tl.rsqrt(mean + eps)  # [BLOCK_M]
    y = x * inv_rms[:, None] * W_ptr[offs_n]  # weight is uniform per column
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# Half rotation kernel: for tensor with last 64 dims, rotate (q1, q2) -> (q2, -q1) with cos/sin
@triton.jit
def half_rotate_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, D,
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    for k0 in range(0, D, BLOCK_D):
        d_offs = k0 + offs_d
        mask = (offs_m[:, None] < M) & (d_offs[None, :] < D)
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + d_offs[None, :] * stride_xd,
            mask=mask, other=0.0
        )
        q1 = x[:, :64]
        q2 = x[:, 64:]
        cos_vec = tl.load(Cos_ptr + tl.arange(0, 64), mask=None, other=1.0)
        sin_vec = tl.load(Sin_ptr + tl.arange(0, 64), mask=None, other=0.0)
        # Rotate q1, q2
        q2_rot = q2 * cos_vec[None, :] - q1 * sin_vec[None, :]
        q1_rot = q1 * cos_vec[None, :] + q2 * sin_vec[None, :]
        y = tl.concatenate([q2_rot, q1_rot], axis=1)
        tl.store(
            Y_ptr + offs_m[:, None] * stride_ym + d_offs[None, :] * stride_yd,
            y,
            mask=mask
        )


# Compute attention scores (Q @ K^T) for all (b,h) across all query positions
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Scores_ptr,
    B, H_q, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_ss, stride_sd,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h = pid_bh % H_q

    base_q = Q_ptr + b * stride_qb + h * stride_qh
    base_k = K_ptr + b * stride_kb + h * stride_kh
    base_s = Scores_ptr + b * stride_sb + h * stride_sh

    for i0 in range(0, S, BLOCK_S):
        offs_i = i0 + tl.arange(0, BLOCK_S)
        logits = tl.zeros((BLOCK_S,), dtype=tl.float32)
        scaling = 1.0 / tl.sqrt(D)
        for j0 in range(0, S, BLOCK_S):
            offs_j = j0 + tl.arange(0, BLOCK_S)
            # Load Q rows i
            q_rows = tl.load(
                base_q + offs_i[:, None] * stride_qs + tl.arange(0, D)[None, :] * stride_qd,
                mask=(offs_i[:, None] < S),
                other=0.0
            )  # [BLOCK_S, D]
            # Load K rows j
            k_rows = tl.load(
                base_k + offs_j[None, :] * stride_ks + tl.arange(0, D)[:, None] * stride_kd,
                mask=(offs_j[None, :] < S),
                other=0.0
            )  # [D, BLOCK_S]
            logits += tl.sum(q_rows * k_rows, axis=1) * scaling
        # Store scores for this tile
        tl.store(
            base_s + offs_i * stride_ss + tl.arange(0, D) * stride_sd,
            logits[:, None],
            mask=(offs_i[None, :] < S)
        )


# Causal mask: upper triangular (i < j) -> -inf; otherwise 0
@triton.jit
def causal_mask_kernel(
    Mask_ptr, S,
    stride_mb, stride_mh, stride_ms, stride_md,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # per (b, h) launch; we can ignore b,h here
    for i0 in range(0, S, BLOCK_S):
        offs_i = i0 + tl.arange(0, BLOCK_S)
        for j0 in range(0, S, BLOCK_S):
            offs_j = j0 + tl.arange(0, BLOCK_S)
            # Fill mask with -inf where i < j
            # We write a 2D tile: rows i, cols j
            mask2d = (offs_i[None, :] < S) & (offs_j[:, None] < S)
            # Compute upper-triangular condition: i < j
            triu_cond = (offs_i[None, :]) < (offs_j[:, None])
            # Create -inf tensor
            neg_inf = -float('inf')
            # Store -inf where condition holds
            tl.store(
                Mask_ptr + pid_bh * stride_mb + offs_i[None, :] * stride_ms + offs_j[:, None] * stride_md,
                neg_inf,
                mask=mask2d & triu_cond
            )


# Softmax over a row vector (Scores_ptr[b,h,i,:]) for each i
@triton.jit
def softmax_row_kernel(
    Scores_ptr, Soft_ptr,
    B, H_q, S, D,
    stride_sb, stride_sh, stride_ss, stride_sd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # one program per (b,h)
    pid_i = tl.program_id(1)  # one program per i
    b = pid_bh // H_q
    h = pid_bh % H_q
    i = pid_i
    # Load row i scores
    scores = tl.load(
        Scores_ptr + b * stride_sb + h * stride_sh + i * stride_ss + tl.arange(0, D) * stride_sd,
        mask=(i < S),
        other=0.0
    )  # [D]
    # Compute softmax: subtract max, exp, sum, normalize
    m = tl.max(scores, axis=0)
    scores = scores - m
    exp_scores = tl.exp(scores)
    denom = tl.sum(exp_scores, axis=0)
    soft = exp_scores / denom
    tl.store(
        Soft_ptr + b * stride_ob + h * stride_oh + i * stride_os + tl.arange(0, D) * stride_od,
        soft,
        mask=(i < S)
    )


# Attention output accumulation: Out[b,h,i] = sum_j Soft[b,h,i,j] * V[b,h,j]
@triton.jit
def attn_out_kernel(
    Soft_ptr, V_ptr, Out_ptr,
    B, H_q, S, D,
    stride_sb, stride_sh, stride_ss, stride_sd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # one program per (b,h)
    b = pid_bh // H_q
    h = pid_bh % H_q
    base_soft = Soft_ptr + b * stride_sb + h * stride_sh
    base_v = V_ptr + b * stride_vb + h * stride_vh
    base_out = Out_ptr + b * stride_ob + h * stride_oh

    for i0 in range(0, S, BLOCK_S):
        offs_i = i0 + tl.arange(0, BLOCK_S)
        out_vec = tl.zeros((BLOCK_S,), dtype=tl.float32)
        # Accumulate over keys
        for j0 in range(0, S, BLOCK_S):
            offs_j = j0 + tl.arange(0, BLOCK_S)
            # Load softmax for each i in tile and V for each j in tile
            soft_mat = tl.load(
                base_soft + offs_i[:, None] * stride_ss + offs_j[None, :] * stride_sd,
                mask=(offs_i[:, None] < S) & (offs_j[None, :] < S),
                other=0.0
            )  # [BLOCK_S, BLOCK_S]
            v_tile = tl.load(
                base_v + offs_j * stride_vs + tl.arange(0, D)[None, :] * stride_vd,
                mask=(offs_j[None, :] < S),
                other=0.0
            )  # [BLOCK_S, D]
            # out[i] += sum_j soft[i,j] * V[j]
            out_vec += tl.sum(soft_mat * v_tile, axis=1)
        # Store results
        tl.store(
            base_out + offs_i * stride_os + tl.arange(0, D) * stride_od,
            out_vec[:, None],
            mask=(offs_i[None, :] < S)
        )


# Final output projection: Out[M, OUT_N] = Attn[M, IN_N] @ OUT_W[OUT_N, IN_N]^T (no bias)
@triton.jit
def linear_out_kernel(
    Attn_ptr, OUT_W_ptr, OUT_ptr,
    M, IN_N, OUT_N,
    stride_am, stride_an,
    stride_wm, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, IN_N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            Attn_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < IN_N),
            other=0.0
        )
        w = tl.load(
            OUT_W_ptr + offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk,
            mask=(offs_n[None, :] < OUT_N) & (offs_k[:, None] < IN_N),
            other=0.0
        )
        acc += tl.dot(a, w)
    tl.store(
        OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_N)
    )


class ModelNew(nn.Module):
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
        rms_norm_eps: float,
    ):
        # Shapes
        B, S, H = hidden_states.shape
        D = 128
        H_q = 96
        H_kv = 8
        G = 12  # num_key_value_groups

        # 1) Linear projections for Q, K, V
        # Allocate outputs
        Q = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        K = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        V = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch linear_fwd_kernel for Q, K, V
        # Grid: (ceil_div(S*H_q, BLOCK_M),)
        grid_linear = (triton.cdiv(S * H_q, 128),)
        linear_fwd_kernel[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            S * H, D, H,  # M, N, K
            S, D,  # stride_xm, stride_xk
            H_q, D,  # stride_wn, stride_wk (note: weights are [N, K], here N=H_q, K=H)
            S, D,  # stride_ym, stride_yn
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        linear_fwd_kernel[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, K,
            S * H, D, H,
            S, D,
            H_q, D,
            S, D,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        linear_fwd_kernel[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, V,
            S * H, D, H,
            S, D,
            H_q, D,
            S, D,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)
        grid_rms = (triton.cdiv(S * H_q, 128),)
        rmsnorm_kernel[grid_rms](
            Q, q_norm_weight, Q_norm,
            S * H_q, D,
            S, D,
            S, D,
            rms_norm_eps,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        rmsnorm_kernel[grid_rms](
            K, k_norm_weight, K_norm,
            S * H_q, D,
            S, D,
            S, D,
            rms_norm_eps,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # 3) Half rotation for Q and K (rotate halves with cos/sin)
        # Load cos/sin; original code uses ones/zeros. We provide them as tensors of proper size.
        cos_vec = cos  # [64]
        sin_vec = sin  # [64]
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)
        grid_half = (triton.cdiv(S * H_q, 128),)
        half_rotate_kernel[grid_half](
            Q_norm, cos_vec, sin_vec, Q_rot,
            S * H_q, D,
            S, D,
            S, D,
            BLOCK_M=128, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        half_rotate_kernel[grid_half](
            K_norm, cos_vec, sin_vec, K_rot,
            S * H_q, D,
            S, D,
            S, D,
            BLOCK_M=128, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 4) Grouped Query Attention: expand K/V to 96 heads (repeat each KV head 12 times)
        # Note: The original code maps each query head to its nearest key head via division by G.
        # We replicate this behavior by mapping:
        # kv_h = q_h // G, then repeat along seq dimension. Here, since we already have 96 heads,
        # mapping is simply identity for kv_h. To ensure GQA, we create [B, H_q, S, D] as is.
        # Expand keys/values to match query heads: K/V already [B, H_q, S, D], V is used per head.

        # 5) Compute attention scores (Q @ K^T) using Triton kernel for each (b, h)
        # Allocate scores [B, H_q, S, D]
        scores = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_scores = (B * H_q,)
        attn_scores_kernel[grid_scores](
            Q_rot, K_rot, scores,
            B, H_q, S, D,
            S, H_q, S, D,
            S, H_q, S, D,
            S, H_q, S, D,
            BLOCK_S=64, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 6) Apply causal mask in Triton
        causal_mask = torch.empty((B, H_q, S, S), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_mask = (B * H_q,)
        causal_mask_kernel[grid_mask](
            causal_mask, S,
            S, H_q, S, S,
            BLOCK_S=64, BLOCK_D=128,
            num_warps=4, num_stages=2
        )
        # For now, we need scores + mask to softmax. Since Triton softmax row kernel is defined,
        # we can compute softmax per i for each (b,h) using scores.

        # 7) Softmax per row: Triton softmax_row_kernel
        # Output Soft: [B, H_q, S, D]
        Soft = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_softmax = (B * H_q, S)
        softmax_row_kernel[grid_softmax](
            scores, Soft,
            B, H_q, S, D,
            S, H_q, S, D,
            S, H_q, S, D,
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 8) Compute attention output: Out[b,h,i] = sum_j Soft[b,h,i,j] * V[b,h,j]
        attn_out = torch.empty((B, H_q, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_out = (B * H_q,)
        attn_out_kernel[grid_out](
            Soft, V, attn_out,
            B, H_q, S, D,
            S, H_q, S, D,
            S, H_q, S, D,
            S, H_q, S, D,
            BLOCK_S=64, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 9) Final output projection
        output = torch.empty((B, S, H_q * D), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_outproj = (triton.cdiv(S * (H_q * D), 128),)
        linear_out_kernel[grid_outproj](
            attn_out, o_proj_weight, output,
            S * (H_q * D), D, H_q * D,
            S, D,
            (H_q * D), D,
            S, (H_q * D),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
