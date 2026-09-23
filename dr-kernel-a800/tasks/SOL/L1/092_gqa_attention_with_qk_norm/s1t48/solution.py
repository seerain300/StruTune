import torch
import triton
import triton.language as tl


@triton.jit
def linear_fwd_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a [BLOCK_M, BLOCK_N] tile of Y
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load X tile [BLOCK_M, BLOCK_K]
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        # Load W tile [BLOCK_N, BLOCK_K]
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0
        )
        # Accumulate
        acc += tl.dot(x, tl.trans(w))

    # Add bias
    bias = tl.load(BIAS_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr, WEIGHT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Normalize each row of X across N with per-row weight
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # typically 128

    # Compute per-row mean of squares across N
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n_chunk = n0 + offs_n
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_n_chunk[None, :] * stride_xn,
            mask=(offs_m[:, None] < M) & (offs_n_chunk[None, :] < N),
            other=0.0
        )  # [BLOCK_M, BLOCK_N]
        sum_sq += tl.sum(x * x, axis=1)

    mean_sq = sum_sq / N
    scale = tl.rsqrt(mean_sq + eps)  # [BLOCK_M]

    # Apply weight and store
    for n0 in range(0, N, BLOCK_N):
        offs_n_chunk = n0 + offs_n
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_n_chunk[None, :] * stride_xn,
            mask=(offs_m[:, None] < M) & (offs_n_chunk[None, :] < N),
            other=0.0
        )
        w = tl.load(
            WEIGHT_ptr + offs_n_chunk * stride_wn,
            mask=offs_n_chunk < N, other=1.0
        )  # [BLOCK_N]
        y = x * scale[:, None] * w[None, :]
        tl.store(
            Y_ptr + offs_m[:, None] * stride_ym + offs_n_chunk[None, :] * stride_yn,
            y,
            mask=(offs_m[:, None] < M) & (offs_n_chunk[None, :] < N)
        )


@triton.jit
def apply_half_rotation_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, N,  # N is the last dimension, expected 128
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles a tile of M rows over N columns
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )

    q1 = x[:, :64]
    q2 = x[:, 64:128]

    c = tl.load(Cos_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]
    s = tl.load(Sin_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]

    q2_rot = q2 * c[None, :] - q1 * s[None, :]
    q1_rot = q2 * s[None, :] + q1 * c[None, :]

    y = tl.concatenate([q2_rot, q1_rot], axis=1)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, D,  # Q,K,V shapes: [B, S, D]
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    scaling: tl.constexpr,
):
    # Each program handles one (batch b, query position i)
    pid = tl.program_id(0)
    b = pid // S
    i = pid % S

    # Compute attention scores for all j in [0, S)
    logits = tl.zeros((S,), dtype=tl.float32)
    for k0 in range(0, S, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load q[i, :] tile
        q = tl.load(Q_ptr + (b * stride_qm + i * stride_qk), mask=None, other=0.0)  # scalar load
        # Load k[j, :] tile
        k = tl.load(K_ptr + (b * stride_km + offs_k * stride_kk), mask=(offs_k < S), other=0.0)  # [BLOCK_K]
        # Accumulate logits
        logits += q * k  # scalar times vector -> vector

    # Apply scaling
    logits = logits * scaling

    # Apply causal mask: j > i -> -inf
    # Triton allows elementwise where on vectors
    causal_mask = (offs_k[None, :] > i)
    logits = tl.where(causal_mask, -float('inf'), logits)

    # Softmax over j
    exp_logits = tl.exp(logits - tl.max(logits))  # stabilize by max
    denom = tl.sum(exp_logits)
    softmax = exp_logits / denom

    # Accumulate output with V
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for v0 in range(0, S, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        v = tl.load(V_ptr + (b * stride_vm + offs_v * stride_vk), mask=(offs_v < S), other=0.0)  # [BLOCK_V]
        out_vec += tl.sum(softmax[None, :] * v[None, :], axis=1)  # [BLOCK_V] but Triton doesn't support axis=1; sum per v element

    # Triton needs explicit sum across softmax per v element
    for v_idx in range(0, BLOCK_V):
        v_val = v[v_idx]
        out_vec += softmax[offs_v0 + v_idx] * v_val

    # Store out_vec to Out[b, i, :]
    tl.store(Out_ptr + (b * stride_om + i * stride_ok), out_vec)


@triton.jit
def linear_out_kernel(
    Attn_ptr, OUT_W_ptr, OUT_ptr,
    M, IN_N, OUT_N,
    stride_am, stride_an,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a [BLOCK_M, BLOCK_N] tile of OUT
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IN_N
    for k0 in range(0, IN_N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a = tl.load(
            Attn_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < IN_N),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]

        w = tl.load(
            OUT_W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wm,
            mask=(offs_n[:, None] < OUT_N) & (offs_k[None, :] < IN_N),
            other=0.0
        )  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(a, tl.trans(w))  # [BLOCK_M, BLOCK_N]

    tl.store(
        OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_N)
    )


class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads: int = 96, num_key_value_heads: int = 8, num_key_value_groups: int = 12, head_dim: int = 128):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        # For this evaluation, we will assume hidden_states is [B, S, H], where H is not used directly in the heavy path.
        # We only need to compute Q, K, V with the provided weights. Bias is provided.

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
        B, S, H = hidden_states.shape  # [batch, seq_len, hidden_dim]
        D = self.head_dim  # 128
        H_q = self.num_attention_heads  # 96
        H_k = self.num_key_value_heads  # 8
        groups = self.num_key_value_groups  # 12 (use 12 to avoid runtime errors)

        # 1) Compute Q, K, V via Triton linear
        # Note: hidden_states shape is [B, S, H]; for linear, X has shape [M, K], Y has shape [M, N].
        # We need to pass K and N properly. In the original, hidden_states is the input to linear for Q, K, V.
        # Here, we treat hidden_states as X with K=H, N=H_q*D or H_k*D accordingly.
        Q = torch.empty((B, S, H_q * D), device=hidden_states.device, dtype=hidden_states.dtype)
        K = torch.empty((B, S, H_k * D), device=hidden_states.device, dtype=hidden_states.dtype)
        V = torch.empty((B, S, H_k * D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton linear kernels
        # For Q: M=B*S, K=H, N=H_q*D
        linear_fwd_kernel[(B, S,)](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B * S, H_q * D, H,
            hidden_states.stride(0), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        # For K: M=B*S, K=H, N=H_k*D
        linear_fwd_kernel[(B, S,)](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B * S, H_k * D, H,
            hidden_states.stride(0), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2
        )

        # For V: M=B*S, K=H, N=H_k*D
        linear_fwd_kernel[(B, S,)](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B * S, H_k * D, H,
            hidden_states.stride(0), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        rmsnorm_kernel[(B, S,)](
            Q, Q_norm, q_norm_weight,
            B * S, H_q * D,
            Q.stride(0), Q.stride(1),
            q_norm_weight.stride(0), q_norm_weight.stride(1),
            Q_norm.stride(0), Q_norm.stride(1),
            rms_norm_eps,
            BLOCK_M=32, BLOCK_N=128, num_warps=4, num_stages=2
        )

        rmsnorm_kernel[(B, S,)](
            K, K_norm, k_norm_weight,
            B * S, H_k * D,
            K.stride(0), K.stride(1),
            k_norm_weight.stride(0), k_norm_weight.stride(1),
            K_norm.stride(0), K_norm.stride(1),
            rms_norm_eps,
            BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=2
        )

        # 3) Apply half rotation to Q and K
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        apply_half_rotation_kernel[(B, S,)](
            Q_norm, cos, sin, Q_rot,
            B * S, H_q * D,
            Q_norm.stride(0), Q_norm.stride(1),
            Q_rot.stride(0), Q_rot.stride(1),
            BLOCK_M=32, BLOCK_N=128, num_warps=4, num_stages=2
        )

        apply_half_rotation_kernel[(B, S,)](
            K_norm, cos, sin, K_rot,
            B * S, H_k * D,
            K_norm.stride(0), K_norm.stride(1),
            K_rot.stride(0), K_rot.stride(1),
            BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=2
        )

        # 4) Grouped Query Attention expansion to [B, H_q, S, D]
        # Original uses H_k=8, H_q=96, groups=12
        K_rot_expanded = K_rot.view(B, S, H_k, 1, D).repeat_interleave(groups, dim=2)  # [B, S, H_q, 1, D]
        V_expanded = V.view(B, S, H_k, 1, D).repeat_interleave(groups, dim=2)  # [B, S, H_q, 1, D]
        K_expanded = K_rot_expanded.view(B, S, H_q, D)  # [B, S, H_q, D]
        V_expanded = V_expanded.view(B, S, H_q, D)      # [B, S, H_q, D]

        # Make contiguous
        K_expanded = K_expanded.contiguous()
        V_expanded = V_expanded.contiguous()

        # 5) Compute attention via Triton kernel: Out[b, i, :] = sum_j softmax_j(q[i, :] @ k[j, :]) * v[j, :]
        # We need Q_rot shaped as [B, S, D*H_q] for kernel (already [B, S, H_q*D])
        Out = torch.empty((B, S, D * H_q), device=hidden_states.device, dtype=hidden_states.dtype)

        # scaling = 1 / sqrt(D)
        scaling = 1.0 / (D ** 0.5)

        attention_kernel[(B * S,)](
            Q_rot, K_expanded, V_expanded, Out,
            B, S, D * H_q,
            Q_rot.stride(0), Q_rot.stride(2),
            K_expanded.stride(0), K_expanded.stride(2),
            V_expanded.stride(0), V_expanded.stride(2),
            Out.stride(0), Out.stride(2),
            BLOCK_M=1, BLOCK_K=64, BLOCK_V=64, scaling=scaling,
            num_warps=4, num_stages=2
        )

        # 6) Final output projection via Triton: Out_proj
        Attn = Out  # after attention we have [B, S, H_q*D]
        Output = torch.empty((B, S, o_proj_weight.shape[0]), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch final linear
        M = B * S
        IN_N = H_q * D
        OUT_N = o_proj_weight.shape[0]  # typically 3072
        linear_out_kernel[(B, S,)](
            Attn, o_proj_weight, Output,
            M, IN_N, OUT_N,
            Attn.stride(0), Attn.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1),
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2
        )

        return Output


def run(*args):
    return ModelNew()(*args)
