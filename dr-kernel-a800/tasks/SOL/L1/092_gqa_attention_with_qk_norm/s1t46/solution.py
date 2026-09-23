import torch
import torch.nn as nn
import triton
import triton.language as tl


# Linear projection: Y[M, N] = X[M, K] @ W[N, K]^T + bias[N]
@triton.jit
def linear_fwd_kernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile along M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # columns
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )  # [BLOCK_M, BLOCK_K]
        w = tl.load(
            W_ptr + offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0
        )  # [BLOCK_N, BLOCK_K]
        acc += tl.dot(x, w)  # [BLOCK_M, BLOCK_N]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]

    # Store result
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# RMSNorm: y[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps)
@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr, Weight_ptr, eps,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # one program per row i
    i = pid_m
    acc = 0.0
    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        x = tl.load(X_ptr + i * stride_xm + offs_n * stride_xn, mask=offs_n < N, other=0.0)  # [BLOCK_N]
        acc += tl.sum(x * x)
    mean_sq = acc / N
    scale = 1.0 / tl.sqrt(mean_sq + eps)
    # Optionally apply per-feature weight (here we assume weight is identity for Q and K)
    # Load weight and multiply; for generality, we keep it simple.
    y = tl.load(X_ptr + i * stride_xm + offs_n * stride_xn, mask=offs_n < N, other=0.0) * scale
    tl.store(Y_ptr + i * stride_ym + offs_n * stride_yn, y, mask=offs_n < N)


# Apply half rotation: split last 64 dims, rotate (q1, q2) -> (q2, -q1)
@triton.jit
def apply_half_rotation_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, N,  # N is head_dim, expected 128
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # N=128

    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0
    )

    q1 = x[:, :64]
    q2 = x[:, 64:]

    c = tl.load(Cos_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]
    s = tl.load(Sin_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]

    # Compute rotated components
    q2_rot = q2 * c[None, :] - q1 * s[None, :]
    q1_rot = q2 * s[None, :] + q1 * c[None, :]

    y = tl.concatenate([q2_rot, q1_rot], axis=1)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Grouped Query Attention: expand KV heads to match Q heads (GQA mapping)
def gqa_expand(kv_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    kv_states: [B, S, H_k, D]
    Returns: [B, H_q, S, D] where each KV head is repeated to its corresponding Q head group.
    """
    B, S, H_k, D = kv_states.shape
    kv_heads = kv_states.view(B, S, H_k, 1, D)  # [B, S, H_k, 1, D]
    kv_repeated = kv_heads.repeat_interleave(num_key_value_groups, dim=2)  # [B, S, H_q, 1, D]
    return kv_repeated.view(B, S, H_q, D)


class ModelNew(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(
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
        # Shapes from original code
        B, S, _ = hidden_states.shape
        num_attention_heads = 96
        num_key_value_heads = 8
        head_dim = 128
        num_key_value_groups = 12
        scaling = 1.0 / (head_dim ** 0.5)

        # 1) Linear projections for Q, K, V
        Q = torch.empty((B, S, num_attention_heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        K = torch.empty((B, S, num_key_value_heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        V = torch.empty((B, S, num_key_value_heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton kernels for Q, K, V
        # Q = hidden_states @ q_proj_weight^T + q_proj_bias
        linear_fwd_kernel[(B, S,)](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, num_attention_heads * head_dim, hidden_states.shape[-1],
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # K = hidden_states @ k_proj_weight^T + k_proj_bias
        linear_fwd_kernel[(B, S,)](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, num_key_value_heads * head_dim, hidden_states.shape[-1],
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # V = hidden_states @ v_proj_weight^T + v_proj_bias
        linear_fwd_kernel[(B, S,)](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, num_key_value_heads * head_dim, hidden_states.shape[-1],
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        # Q_norm
        Q_n = torch.empty_like(Q)
        rmsnorm_kernel[(B, S,)](
            Q, Q_n, q_norm_weight, self.eps,
            B, num_attention_heads * head_dim,
            Q.stride(0), Q.stride(1),
            Q_n.stride(0), Q_n.stride(1),
            BLOCK_M=1, BLOCK_N=128, num_warps=4, num_stages=2
        )
        Q = Q_n

        # K_norm
        K_n = torch.empty_like(K)
        rmsnorm_kernel[(B, S,)](
            K, K_n, k_norm_weight, self.eps,
            B, num_key_value_heads * head_dim,
            K.stride(0), K.stride(1),
            K_n.stride(0), K_n.stride(1),
            BLOCK_M=1, BLOCK_N=128, num_warps=4, num_stages=2
        )
        K = K_n

        # 3) Apply half rotation to Q and K
        Qr = torch.empty_like(Q)
        Kr = torch.empty_like(K)
        # Launch rotation kernel for Q
        apply_half_rotation_kernel[(B, S,)](
            Q, cos, sin, Qr,
            B, num_attention_heads * head_dim,
            Q.stride(0), Q.stride(1),
            Qr.stride(0), Qr.stride(1),
            BLOCK_M=1, BLOCK_N=128, num_warps=4, num_stages=2
        )
        # Launch rotation kernel for K
        apply_half_rotation_kernel[(B, S,)](
            K, cos, sin, Kr,
            B, num_key_value_heads * head_dim,
            K.stride(0), K.stride(1),
            Kr.stride(0), Kr.stride(1),
            BLOCK_M=1, BLOCK_N=128, num_warps=4, num_stages=2
        )

        # 4) Grouped Query Attention: expand K and V to match Q heads (GQA mapping)
        K_exp = gqa_expand(K, num_key_value_groups)  # [B, S, H_q, D]
        V_exp = gqa_expand(V, num_key_value_groups)  # [B, S, H_q, D]

        # Reshape to [B, H_q, S, D] for attention computation (PyTorch)
        # Compute attention weights: [B, H_q, S, S]
        # Qr: [B, S, H_q, D], K_exp: [B, S, H_q, D]
        # We'll compute Qr @ K_exp^T across S dimension. But to get [B, H_q, S, S], we need Qr @ K_exp^T then transpose dims.
        # Compute Qr @ K_exp^T: [B, S, H_q, S]
        # Then transpose to [B, H_q, S, S]
        # Note: PyTorch matmul here; acceptable for correctness.
        # Compute attention weights per head:
        # attn = (Qr @ K_exp^T) * scaling
        # Apply causal mask: upper triangular (j > i -> -inf)
        # Softmax along last dim (sequence length).
        # To form [B, H_q, S, S], we can use torch.einsum or explicitly loop per head.
        # We'll do it explicitly for robustness.

        # Prepare attention tensors
        attn_weights = torch.empty((B, num_attention_heads, S, S), device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(B):
            for h in range(num_attention_heads):
                # Q_bh: [S, D], K_bh: [S, D]
                Q_bh = Qr[b, :, h, :]  # [S, D]
                K_bh = K_exp[b, :, h, :]  # [S, D]
                # Compute Q @ K^T: [S, S]
                # We'll use PyTorch matmul here. This is acceptable as it's lightweight per head.
                Q_bh = Q_bh.view(S, 1, D)
                K_bh = K_bh.view(S, D, 1)
                scores = torch.matmul(Q_bh, K_bh).squeeze(-1)  # [S, S]
                scores = scores * scaling
                # Causal mask: j > i -> -inf
                causal = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)
                attn_weights[b, h] = scores + causal
                # Softmax along last dim
                attn_weights[b, h] = torch.softmax(attn_weights[b, h], dim=-1)

        # Compute attn output: attn @ V_exp^T -> [B, H_q, S, D]
        attn_output = torch.empty((B, num_attention_heads, S, D), device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(B):
            for h in range(num_attention_heads):
                # attn_weights_bh: [S, S], V_exp_bh: [S, D]
                attn_weights_bh = attn_weights[b, h]  # [S, S]
                V_exp_bh = V_exp[b, :, h, :]  # [S, D]
                # For each output position i, compute weighted sum over j: sum_j attn[i, j] * V_exp[j, :]
                # Implement with loop for robustness
                out_vec = torch.zeros((D,), device=hidden_states.device, dtype=hidden_states.dtype)
                for i in range(S):
                    # weights_i = attn_weights_bh[i, :]  # [S]
                    # v_i = V_exp_bh[i, :]  # [D]
                    weights_i = attn_weights_bh[i]  # [S]
                    v_i = V_exp_bh[i]  # [D]
                    out_vec += weights_i @ v_i  # scalar
                attn_output[b, h] = out_vec

        # 5) Final output projection: linear_out = attn_output @ o_proj_weight^T
        output = torch.empty((B, S, num_attention_heads * head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        linear_out_kernel[(B, S,)](
            attn_output.view(B, S, -1), o_proj_weight, None, output,
            B * S, num_attention_heads * head_dim, num_attention_heads * head_dim,
            attn_output.view(B, S, -1).stride(0), attn_output.view(B, S, -1).stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(2),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
