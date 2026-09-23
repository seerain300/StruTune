import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_fwd_kernel(
    X_ptr,      # *ptr to [M, K]
    W_ptr,      # *ptr to [N, K]
    BIAS_ptr,   # *ptr to [N] or None
    Y_ptr,      # *ptr to [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)

        acc += tl.dot(x, w)

    if has_bias:
        bias = tl.load(BIAS_ptr + offs_n, mask=(offs_n < N), other=0.0)
        acc += bias[None, :]

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *ptr to [M, N], row-major
    W_ptr,        # *ptr to [N], scale (usually 1)
    Y_ptr,        # *ptr to [M, N], output
    M: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_w,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m  # one program per row
    sumsq = 0.0
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(X_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=(offs_n < N), other=0.0)
        sumsq += tl.sum(x * x)

    inv_rms = 1.0 / tl.sqrt(sumsq / N + eps)

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(X_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=(offs_n < N), other=0.0)
        w = tl.load(W_ptr + offs_n * stride_w, mask=(offs_n < N), other=1.0)
        y = x * inv_rms * w
        tl.store(Y_ptr + offs_m * stride_ym + offs_n * stride_yn, y, mask=(offs_n < N))


@triton.jit
def apply_half_rotation_kernel(
    X_ptr,       # [M, N], input
    Y_ptr,       # [M, N], output
    M: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    COS_PTR, SIN_PTR, HALF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load block
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # Half rotation: split into first and second halves of last dimension
    # Note: offs_n ranges over N=128, HALF=64, so we can split safely
    q1 = x[:, :HALF]
    q2 = x[:, HALF:]
    cos_vec = tl.load(COS_PTR + tl.arange(0, HALF), mask=True, other=1.0)  # [HALF]
    sin_vec = tl.load(SIN_PTR + tl.arange(0, HALF), mask=True, other=0.0)  # [HALF]

    # Combine rotated q2 with original q1
    q_rot_half = q2 * cos_vec[None, :] + (-(q1) * sin_vec[None, :])
    y = tl.zeros_like(x)
    y[:, :HALF] = q_rot_half
    y[:, HALF:] = q1

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def linear_out_kernel(
    X_ptr,      # *ptr to [M, K] (attention output)
    W_ptr,      # *ptr to [N, K]
    Y_ptr,      # *ptr to [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)

        acc += tl.dot(x, w)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fix constants from original code
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = 12  # since 8 * 12 = 96

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight,
                cos: torch.Tensor, sin: torch.Tensor):
        """
        hidden_states: [B, S, 768]
        q_proj_weight, k_proj_weight, v_proj_weight: [768, 768]
        q_proj_bias, k_proj_bias, v_proj_bias: [768]
        o_proj_weight: [768, 768]
        q_norm_weight, k_norm_weight: [96*128], [8*128]
        cos, sin: [128], rotation on last 64 dims
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        B, S, hidden_dim = hidden_states.shape
        assert hidden_dim == 768, "hidden_dim must be 768"

        HALF = 64
        D = self.head_dim
        eps = 1e-8  # RMSNorm epsilon

        # 1) Flatten hidden states for linear projection
        hs_flat = hidden_states.reshape(B * S, hidden_dim).contiguous()

        # 2) Compute Q, K, V via Triton linear projection (fp32 compute)
        query = torch.empty((B * S, hidden_dim), device=device, dtype=torch.float32)
        key = torch.empty((B * S, hidden_dim), device=device, dtype=torch.float32)
        value = torch.empty((B * S, hidden_dim), device=device, dtype=torch.float32)

        grid_q = (triton.cdiv(B * S, 128), triton.cdiv(hidden_dim, 128))
        linear_fwd_kernel[grid_q](
            hs_flat, q_proj_weight, q_proj_bias, query,
            B * S, hidden_dim, hidden_dim,
            hs_flat.stride(0), hidden_dim,
            q_proj_weight.stride(0), hidden_dim,
            query.stride(0), hidden_dim,
            has_bias=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        grid_k = (triton.cdiv(B * S, 128), triton.cdiv(hidden_dim, 128))
        linear_fwd_kernel[grid_k](
            hs_flat, k_proj_weight, k_proj_bias, key,
            B * S, hidden_dim, hidden_dim,
            hs_flat.stride(0), hidden_dim,
            k_proj_weight.stride(0), hidden_dim,
            key.stride(0), hidden_dim,
            has_bias=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        grid_v = (triton.cdiv(B * S, 128), triton.cdiv(hidden_dim, 128))
        linear_fwd_kernel[grid_v](
            hs_flat, v_proj_weight, v_proj_bias, value,
            B * S, hidden_dim, hidden_dim,
            hs_flat.stride(0), hidden_dim,
            v_proj_weight.stride(0), hidden_dim,
            value.stride(0), hidden_dim,
            has_bias=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Apply RMSNorm on Q and K (elementwise, per last dim)
        # Compute Y = X * rsqrt(mean(X^2) + eps), scale by weight
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # For Q
        grid_rn_q = (B * S, triton.cdiv(hidden_dim, 128))
        rmsnorm_kernel[grid_rn_q](
            query, q_norm_weight, query_norm,
            B * S, hidden_dim,
            query.stride(0), hidden_dim,
            query_norm.stride(0), hidden_dim,
            q_norm_weight.stride(0),
            eps,
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # For K
        grid_rn_k = (B * S, triton.cdiv(hidden_dim, 128))
        rmsnorm_kernel[grid_rn_k](
            key, k_norm_weight, key_norm,
            B * S, hidden_dim,
            key.stride(0), hidden_dim,
            key_norm.stride(0), hidden_dim,
            k_norm_weight.stride(0),
            eps,
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # 4) Apply half rotation for Q and K
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        grid_qr = (triton.cdiv(B * S, 128), triton.cdiv(hidden_dim, 128))
        apply_half_rotation_kernel[grid_qr](
            query_norm, query_rot,
            B * S, hidden_dim,
            query_norm.stride(0), hidden_dim,
            query_rot.stride(0), hidden_dim,
            BLOCK_M=128, BLOCK_N=128,
            COS_PTR=cos, SIN_PTR=sin, HALF=HALF,
            num_warps=4, num_stages=2,
        )

        grid_kr = (triton.cdiv(B * S, 128), triton.cdiv(hidden_dim, 128))
        apply_half_rotation_kernel[grid_kr](
            key_norm, key_rot,
            B * S, hidden_dim,
            key_norm.stride(0), hidden_dim,
            key_rot.stride(0), hidden_dim,
            BLOCK_M=128, BLOCK_N=128,
            COS_PTR=cos, SIN_PTR=sin, HALF=HALF,
            num_warps=4, num_stages=2,
        )

        # 5) Reshape Q/K/V to [B, S, H, D] where H = num_attention_heads (96), D = 128
        # Note: We have B*S rows and 768-dim vectors. We need to form 96 heads per row.
        # Each row corresponds to one (batch, seq) pair; we split the 768-dim vector into 96 heads of 128 dims.
        # However, we already have Q/K/V as [B*S, 768] from projection. We proceed to attention.

        # Reconstruct query/key/value as [B, S, H, D] by viewing. Each (b,s) row is 768, split into 96 chunks.
        # We can directly compute attention from [B*S, 768] without explicit 4D view.
        # For attention, we use matmul in PyTorch to ensure correctness.

        # 6) Compute attention scores using PyTorch:
        # Score = (Q @ K^T) * scaling, where scaling = 1 / sqrt(D)
        scaling = 1.0 / (D ** 0.5)
        # Convert to [B*S, 768] and ensure contiguity
        query_t = query_rot.view(B * S, hidden_dim).contiguous()
        key_t = key_rot.view(B * S, hidden_dim).contiguous()
        value_t = value.view(B * S, hidden_dim).contiguous()

        scores = torch.matmul(query_t, key_t.transpose(0, 1)) * scaling  # [B*S, B*S]
        # Apply causal mask: upper-triangular within sequence
        seq = query_t.shape[0]
        causal_mask = torch.triu(torch.full((seq, seq), float('-inf'), device=device, dtype=scores.dtype), diagonal=1)
        scores = scores + causal_mask

        # Softmax over sequence dimension
        probs = torch.softmax(scores, dim=-1)  # [B*S, B*S]

        # Compute attention output: O = probs @ value
        attn_output = torch.matmul(probs, value_t)  # [B*S, 768]

        # 7) Final output projection via Triton: Out[M, OUT_N] = Attn[M, K] @ o_proj_weight[OUT_N, K]^T
        out_dim = o_proj_weight.shape[0]  # 768
        output = torch.empty((B * S, out_dim), device=device, dtype=torch.float32)

        grid_out = (triton.cdiv(B * S, 128), triton.cdiv(out_dim, 128))
        linear_out_kernel[grid_out](
            attn_output, o_proj_weight,
            output,
            B * S, out_dim, hidden_dim,  # attn_output is [B*S, 768], o_proj_weight is [768, 768]
            attn_output.stride(0), hidden_dim,
            o_proj_weight.stride(0), hidden_dim,
            output.stride(0), hidden_dim,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 8) Reshape back to [B, S, out_dim] and cast to original dtype if needed
        final = output.view(B, S, out_dim)

        return final


def run(*args):
    return ModelNew()(*args)
