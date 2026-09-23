import torch
import triton
import triton.language as tl


# Linear projection: Y[M, N] = X[M, K] @ W[N, K]^T + bias[N]
@triton.jit
def linear_fused_kernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # row index in M
    if pid >= M:
        return
    # Loop over output columns in tiles
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Reduction over K
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            # Load X row slice: [BLOCK_K]
            x = tl.load(X_ptr + pid * stride_xm + offs_k * stride_xk, mask=(offs_k < K), other=0.0)
            # Load W^T columns: [BLOCK_K, BLOCK_N]
            w = tl.load(W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                        mask=((offs_n[None, :] < N) & (offs_k[:, None] < K)), other=0.0)
            acc += tl.sum(x[:, None] * w, axis=0)  # [BLOCK_N]
        # Add bias
        b = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
        acc = acc + b
        tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, acc, mask=(offs_n < N))


# RMSNorm per row: y[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps)
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_w, stride_y,
    eps,  # float32 scalar
):
    pid = tl.program_id(0)  # row index
    if pid >= M:
        return
    # Load row x
    x = tl.load(X_ptr + pid * stride_xm + tl.arange(0, N) * stride_xn, mask=True, other=0.0).to(tl.float32)
    # Compute variance
    var = tl.sum(x * x, axis=0) / N
    scale = tl.rsqrt(var + eps)  # rsqrt in Triton
    # Normalize and apply weight
    y = (x * scale) * tl.load(Weight_ptr + tl.arange(0, N) * stride_w, mask=True, other=1.0).to(tl.float32)
    tl.store(Y_ptr + pid * stride_y + tl.arange(0, N) * stride_y, y, mask=True)


# Half-rotation for last 128 dims (split 64/64): rotate q1<-q2, q2<--q1, and apply cos/sin to q2
@triton.jit
def half_rotate_kernel_q(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
):
    pid = tl.program_id(0)  # row index
    if pid >= M:
        return
    # N is 128; half = 64
    half = 64
    for n0 in range(0, N, half):
        offs_n = n0 + tl.arange(0, half)
        x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=(offs_n < N), other=0.0).to(tl.float32)
        # q1, q2 split
        q1 = x[:half]
        q2 = x[half:]
        # cos/sin for first half
        c = tl.load(Cos_ptr + tl.arange(0, half), mask=True, other=1.0).to(tl.float32)
        s = tl.load(Sin_ptr + tl.arange(0, half), mask=True, other=1.0).to(tl.float32)
        # rotate q1 and q2: (q1, q2) -> (q2, -q1) rotated by cos/sin
        q2_rot = q2 * c - q1 * s
        q1_rot = q2 * s + q1 * c  # correct rotation
        # Combine back
        y = tl.concatenate([q2_rot, q1_rot], axis=0)
        tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, y, mask=(offs_n < N))


@triton.jit
def half_rotate_kernel_k(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
):
    pid = tl.program_id(0)  # row index
    if pid >= M:
        return
    half = 64
    for n0 in range(0, N, half):
        offs_n = n0 + tl.arange(0, half)
        x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=(offs_n < N), other=0.0).to(tl.float32)
        q1 = x[:half]
        q2 = x[half:]
        c = tl.load(Cos_ptr + tl.arange(0, half), mask=True, other=1.0).to(tl.float32)
        s = tl.load(Sin_ptr + tl.arange(0, half), mask=True, other=1.0).to(tl.float32)
        q2_rot = q2 * c - q1 * s
        q1_rot = q2 * s + q1 * c
        y = tl.concatenate([q2_rot, q1_rot], axis=0)
        tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, y, mask=(offs_n < N))


# Grouped Query Attention:
# For each (batch, query i), compute scores S_i[j] = Q_i · K_j^T * scaling, apply causal mask (j >= i -> -inf), softmax along j, then accumulate O_i = sum_j softmax S_i[j] * V_j.
@triton.jit
def gqa_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, D,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    BLOCK_K: tl.constexpr,  # tile over key dimension
):
    pid = tl.program_id(0)  # (batch, i) program
    if pid >= B * S:
        return
    b = pid // S
    i = pid % S

    # Initialize output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)

    # Compute logits S_i[j] for all j in tiles
    scores = tl.zeros((S,), dtype=tl.float32)
    scaling = 1.0 / tl.sqrt(D)
    for j0 in range(0, S, BLOCK_K):
        offs_j = j0 + tl.arange(0, BLOCK_K)
        # Load Q_i and K_j tiles
        q = tl.load(Q_ptr + (b * stride_qm + i * stride_qk), mask=True, other=0.0).to(tl.float32)  # [D]
        k = tl.load(K_ptr + (b * stride_km + offs_j * stride_kk), mask=(offs_j < S), other=0.0)   # [BLOCK_K, D]
        # Compute dot: scores[offs_j] = sum_k q[k] * k[offs_j, k]
        for kk in range(0, D):
            scores += (q[kk] * tl.load(K_ptr + (b * stride_km + offs_j * stride_kk + kk * stride_kk),
                                        mask=(offs_j < S), other=0.0))
        scores = scores * scaling

        # Apply causal mask: j >= i -> -inf
        scores = tl.where(offs_j >= i, scores, -float('inf'))

        # Softmax over keys in this tile
        max_score = tl.max(scores, axis=0)
        scores = scores - max_score
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        probs = exp_scores / sum_exp

        # Accumulate output with V_j
        for jj in range(0, BLOCK_K):
            j = j0 + jj
            if j < S:
                v = tl.load(V_ptr + (b * stride_vm + j * stride_vk), mask=True, other=0.0).to(tl.float32)  # [D]
                out_vec += probs[jj] * v

    # Store output for (b, i)
    tl.store(Out_ptr + (b * stride_om + i * stride_ok), out_vec, mask=True)


# Final output projection: Out[M, OUT_N] = Attn[M, IN_N] @ OUT_W[OUT_N, IN_N]^T (no bias)
@triton.jit
def linear_out_kernel(
    Attn_ptr, OUT_W_ptr, Out_ptr,
    M, IN_N, OUT_N,
    stride_am, stride_an,
    stride_wm, stride_wk,  # OUT_W is [OUT_N, IN_N]
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    for n0 in range(0, OUT_N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, IN_N, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(Attn_ptr + pid_m * stride_am + offs_k * stride_an,
                        mask=(offs_k < IN_N), other=0.0)
            w = tl.load(OUT_W_ptr + offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk,
                        mask=((offs_n[None, :] < OUT_N) & (offs_k[:, None] < IN_N)),
                        other=0.0)
            acc += tl.sum(a[:, None] * w, axis=0)
        tl.store(Out_ptr + pid_m * stride_om + offs_n * stride_on, acc, mask=(offs_n < OUT_N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We do not rely on any parameters; Triton kernels will handle all compute.
        # The evaluator will pass inputs/weights/bias into forward.

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
        B, S, D = hidden_states.shape
        H_q = 96
        H_kv = 8
        H_g = 12
        Ddim = 128  # head_dim

        # 1) Q, K, V projections via linear_fused_kernel
        # Allocate outputs
        q = torch.empty((B, S, H_q * Ddim), device=hidden_states.device, dtype=hidden_states.dtype)
        k = torch.empty((B, S, H_kv * Ddim), device=hidden_states.device, dtype=hidden_states.dtype)
        v = torch.empty((B, S, H_kv * Ddim), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Q projection
        grid_q = (B * S,)
        linear_fused_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, q,
            B * S, H_q * Ddim, D,
            S, 1,  # stride_xm=seq_len, stride_xk=1 (row-major along last dim assumed contiguous)
            q_proj_weight.shape[0], q_proj_weight.shape[1],
            H_q * Ddim, 1,
            BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )
        # Launch K projection
        grid_k = (B * S,)
        linear_fused_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, k,
            B * S, H_kv * Ddim, D,
            S, 1,
            k_proj_weight.shape[0], k_proj_weight.shape[1],
            H_kv * Ddim, 1,
            BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )
        # Launch V projection
        grid_v = (B * S,)
        linear_fused_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, v,
            B * S, H_kv * Ddim, D,
            S, 1,
            v_proj_weight.shape[0], v_proj_weight.shape[1],
            H_kv * Ddim, 1,
            BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        q_norm = torch.empty_like(q, dtype=torch.float32)
        k_norm = torch.empty_like(k, dtype=torch.float32)

        # Launch RMSNorm for Q
        grid_rms_q = (B * S,)
        rmsnorm_kernel[grid_rms_q](
            q, q_norm_weight, q_norm,
            B * S, H_q * Ddim,
            H_q * Ddim, 1,
            q_norm_weight.numel(), 1,
            rms_norm_eps, num_warps=4, num_stages=2
        )
        # Launch RMSNorm for K
        grid_rms_k = (B * S,)
        rmsnorm_kernel[grid_rms_k](
            k, k_norm_weight, k_norm,
            B * S, H_kv * Ddim,
            H_kv * Ddim, 1,
            k_norm_weight.numel(), 1,
            rms_norm_eps, num_warps=4, num_stages=2
        )

        # 3) Half-rotation for Q and K
        q_rot = torch.empty_like(q_norm)
        k_rot = torch.empty_like(k_norm)

        grid_half_q = (B * S,)
        half_rotate_kernel_q[grid_half_q](
            q_norm, cos, sin, q_rot,
            B * S, H_q * Ddim,
            H_q * Ddim, 1,
            (B * S) * Ddim, 1,
            num_warps=4, num_stages=2
        )

        grid_half_k = (B * S,)
        half_rotate_kernel_k[grid_half_k](
            k_norm, cos, sin, k_rot,
            B * S, H_kv * Ddim,
            H_kv * Ddim, 1,
            (B * S) * Ddim, 1,
            num_warps=4, num_stages=2
        )

        # 4) Grouped Query Attention: compute attn_output using Triton kernel
        # attn_output: [B, S, H_q*Ddim]
        attn_out = torch.empty((B, S, H_q * Ddim), device=hidden_states.device, dtype=torch.float32)

        grid_attn = (B * S,)
        gqa_attention_kernel[grid_attn](
            q_rot, k_rot, v, attn_out,
            B, S, Ddim,
            H_q * Ddim, 1,
            H_kv * Ddim, 1,
            H_kv * Ddim, 1,
            (B * S) * Ddim, 1,
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 5) Final output projection via Triton
        out = torch.empty((B, S, H_q * Ddim), device=hidden_states.device, dtype=torch.float32)

        grid_out = (B * S,)
        linear_out_kernel[grid_out](
            attn_out, o_proj_weight, out,
            B * S, H_q * Ddim, H_q * Ddim,
            S, 1,
            o_proj_weight.shape[0], o_proj_weight.shape[1],
            S, 1,
            BLOCK_M=1, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
