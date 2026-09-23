import torch
import triton
import triton.language as tl


@triton.jit
def linear_fwd_kernel(
    X_ptr,        # *ptr [M, K]
    W_ptr,        # *ptr [N, K]
    B_ptr,        # *ptr [N] or None (bias), pass pointer or None
    Y_ptr,        # *ptr [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute Y[m, n] = sum_k X[m, k] * W[n, k] + (bias[n] if HAS_BIAS)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)

        acc += tl.dot(x, tl.trans(w))

    if HAS_BIAS:
        b_ptrs = B_ptr + offs_n
        b = tl.load(b_ptrs, mask=(offs_n < N), other=0.0)
        acc += b[None, :]

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *ptr [M, N]
    W_ptr,        # *ptr [N] scaling, usually all ones
    Y_ptr,        # *ptr [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_w,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Per-row RMS normalization: Y = X * rsqrt(mean(X^2) + eps)
    pid_m = tl.program_id(0)
    row_x = X_ptr + pid_m * stride_xm
    row_y = Y_ptr + pid_m * stride_ym

    sumsq = 0.0
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(row_x + offs_n * stride_xn, mask=(offs_n < N), other=0.0)
        sumsq += tl.sum(x * x)

    inv_rms = 1.0 / tl.sqrt(sumsq / N + eps)

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(row_x + offs_n * stride_xn, mask=(offs_n < N), other=0.0)
        w = tl.load(W_ptr + offs_n * stride_w, mask=(offs_n < N), other=1.0)
        y = x * inv_rms * w
        tl.store(row_y + offs_n * stride_yn, y, mask=(offs_n < N))


@triton.jit
def apply_half_rotation_kernel(
    X_ptr,        # *ptr [M, N], N must be 128
    COS_ptr,      # *ptr [64]
    SIN_ptr,      # *ptr [64]
    Y_ptr,        # *ptr [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # For each row, split last 64 dims: q1 = x[:64], q2 = x[64:], rotate to (q2, -q1) and combine with cos/sin applied to q2
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for n0 in range(0, N, BLOCK_N):
        row_x = X_ptr + (offs_m[:, None] * stride_xm + (n0 + offs_n)[None, :] * stride_xn)
        x = tl.load(row_x, mask=(offs_m[:, None] < M) & ((n0 + offs_n)[None, :] < N), other=0.0)

        # Extract q1 and q2 halves (assuming N=128)
        q1 = x[:, :64]
        q2 = x[:, 64:]

        # Rotate q1, q2: (q1, q2) -> (q2, -q1)
        new_q1 = q2
        new_q2 = -q1

        # Apply cos/sin to q2's 64 dims
        cos_vec = tl.load(COS_ptr + tl.arange(0, 64))
        sin_vec = tl.load(SIN_ptr + tl.arange(0, 64))
        new_q2 = new_q2 * cos_vec + new_q2 * sin_vec  # broadcast apply

        # Combine back
        rotated = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        rotated[:, :64] = new_q1
        rotated[:, 64:] = new_q2

        y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + (n0 + offs_n)[None, :] * stride_yn)
        tl.store(y_ptrs, rotated, mask=(offs_m[:, None] < M) & ((n0 + offs_n)[None, :] < N))


@triton.jit
def linear_out_kernel(
    Attn_ptr,     # *ptr [M, K]
    OUT_W_ptr,    # *ptr [OUT_N, K]
    OUT_ptr,      # *ptr [M, OUT_N]
    M, K, OUT_N,
    stride_am, stride_ak,
    stride_wm, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Out[M, OUT_N] = Attn[M, K] @ OUT_W[OUT_N, K]^T
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = Attn_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = OUT_W_ptr + (offs_n[:, None] * stride_wm + offs_k[None, :] * stride_wk)
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(a, tl.trans(w))

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads=96, num_key_value_heads=8, head_dim=128, num_key_value_groups=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_key_value_groups

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight,
                cos: torch.Tensor, sin: torch.Tensor, rms_norm_eps: float):
        """
        hidden_states: [B, S, 768]
        q_proj_weight, k_proj_weight, v_proj_weight: [768, 768]
        q_proj_bias, k_proj_bias, v_proj_bias: [768]
        o_proj_weight: [768, 768] (no bias)
        q_norm_weight, k_norm_weight: [num_attention_heads*head_dim], [num_key_value_heads*head_dim]
        cos, sin: [head_dim//2] vectors for rotation
        """
        # Ensure tensors are on CUDA
        device = hidden_states.device

        B, S, hidden_dim = hidden_states.shape
        H_q = self.num_attention_heads
        H_k = self.num_key_value_heads
        D = self.head_dim

        # Flatten [B, S, hidden_dim] to [B*S, hidden_dim]
        hs_flat = hidden_states.reshape(B * S, hidden_dim)

        # 1) Linear Q, K, V via Triton
        M = B * S
        K = hidden_dim
        N = hidden_dim  # output dim for Q/K/V

        query = torch.empty((M, N), device=device, dtype=torch.float32)
        key = torch.empty((M, N), device=device, dtype=torch.float32)
        value = torch.empty((M, N), device=device, dtype=torch.float32)

        # Grid for [M, N]
        grid_linear = (triton.cdiv(M, 128), triton.cdiv(N, 128))

        linear_fwd_kernel[grid_linear](
            hs_flat, q_proj_weight, q_proj_bias, query,
            M, N, K,
            hs_flat.stride(0), hidden_dim,
            q_proj_weight.stride(0), hidden_dim,
            query.stride(0), hidden_dim,
            HAS_BIAS=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        linear_fwd_kernel[grid_linear](
            hs_flat, k_proj_weight, k_proj_bias, key,
            M, N, K,
            hs_flat.stride(0), hidden_dim,
            k_proj_weight.stride(0), hidden_dim,
            key.stride(0), hidden_dim,
            HAS_BIAS=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        linear_fwd_kernel[grid_linear](
            hs_flat, v_proj_weight, v_proj_bias, value,
            M, N, K,
            hs_flat.stride(0), hidden_dim,
            v_proj_weight.stride(0), hidden_dim,
            value.stride(0), hidden_dim,
            HAS_BIAS=True,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 2) RMSNorm for Q and K via Triton (V not normalized)
        # For query (Q), N = H_q * D
        M_q = B * S * H_q
        N_q = H_q * D
        # Reshape query to [B*S, H_q, D] then flatten to [B*S*H_q, D]
        query_heads = query.view(B * S, H_q, D).reshape(M_q, D)
        q_out = torch.empty_like(query_heads, dtype=torch.float32)
        grid_q = (triton.cdiv(M_q, 128), triton.cdiv(D, 128))
        rmsnorm_kernel[grid_q](
            query_heads, q_norm_weight, q_out,
            M_q, D,
            query_heads.stride(0), D,
            q_out.stride(0), D,
            q_norm_weight.stride(0),
            eps=rms_norm_eps,
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        # Reshape back to [B, S, H_q, D]
        query_norm = q_out.view(B, S, H_q, D)

        # For key (K), N = H_k * D
        M_k = B * S * H_k
        N_k = H_k * D
        key_heads = key.view(B * S, H_k, D).reshape(M_k, D)
        k_out = torch.empty_like(key_heads, dtype=torch.float32)
        grid_k = (triton.cdiv(M_k, 128), triton.cdiv(D, 128))
        rmsnorm_kernel[grid_k](
            key_heads, k_norm_weight, k_out,
            M_k, D,
            key_heads.stride(0), D,
            k_out.stride(0), D,
            k_norm_weight.stride(0),
            eps=rms_norm_eps,
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        # Reshape back to [B, S, H_k, D]
        key_norm = k_out.view(B, S, H_k, D)

        # 3) Apply half rotation to Q and K via Triton
        # We need to apply rotation to flattened [M, N] where N=128
        # For Q: [B*S*H_q, D]
        q_flat = query_norm.reshape(M_q, D)
        q_rot = torch.empty_like(q_flat, dtype=torch.float32)
        grid_qrot = (triton.cdiv(M_q, 128), triton.cdiv(D, 128))
        apply_half_rotation_kernel[grid_qrot](
            q_flat, cos, sin, q_rot,
            M_q, D,
            q_flat.stride(0), D,
            q_rot.stride(0), D,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        query_rot = q_rot.view(B, S, H_q, D)

        # For K: [B*S*H_k, D]
        k_flat = key_norm.reshape(M_k, D)
        k_rot = torch.empty_like(k_flat, dtype=torch.float32)
        grid_krot = (triton.cdiv(M_k, 128), triton.cdiv(D, 128))
        apply_half_rotation_kernel[grid_krot](
            k_flat, cos, sin, k_rot,
            M_k, D,
            k_flat.stride(0), D,
            k_rot.stride(0), D,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        key_rot = k_rot.view(B, S, H_k, D)

        # 4) GQA: expand K/V to H_q and reshape
        # Map each query head q_h to kv_h = q_h // num_key_value_groups
        # Repeat across groups
        key_rot_exp = key_rot[:, :, None, :, :].expand(B, H_q, self.num_key_value_groups, S, D).reshape(B, H_q, S, D)
        value_exp = value.view(B, S, H_k, D)[:, :, None, :, :].expand(B, H_q, self.num_key_value_groups, S, D).reshape(B, H_q, S, D)

        # 5) Compute attention weights using PyTorch matmul for reliability (to avoid Triton attention pitfalls)
        # Q @ K^T, scaled by 1/sqrt(D), then causal mask and softmax per (batch, head) row.
        # Q: [B, H_q, S, D], K: [B, H_q, S, D]
        scaling = 1.0 / (D ** 0.5)
        attn_weights = torch.matmul(query_rot.transpose(1, 2), key_rot.transpose(1, 2)) * scaling  # [B, H_q, S, S]
        # Build causal mask (upper-triangular: mask i>=j with -inf)
        seq_len = S
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=attn_weights.dtype), diagonal=1)
        # Add mask per batch
        attn_weights = attn_weights + causal_mask  # broadcasting over batch

        # Softmax along last dim (sequence length)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 6) Output projection via Triton: attn_output [B, H_q, S, D] -> [B, S, H_q*D]
        # attn_output = attn_weights @ value_exp^T
        # We flatten to [B*H_q*S, D] and multiply by OUT_W [768, D]
        attn_flat = attn_weights.reshape(B * H_q * seq_len, D)  # [M2, D]
        output_flat = torch.empty((B * H_q * seq_len, 768), device=device, dtype=torch.float32)

        # Launch linear_out_kernel
        grid_out = (triton.cdiv(B * H_q * seq_len, 128), triton.cdiv(768, 128))
        linear_out_kernel[grid_out](
            attn_flat, o_proj_weight, output_flat,
            B * H_q * seq_len, D, 768,
            attn_flat.stride(0), D,
            o_proj_weight.stride(0), D,
            output_flat.stride(0), 768,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Reshape to [B, S, H_q*D]
        output = output_flat.view(B, seq_len, H_q * D)
        return output


def run(*args):
    return ModelNew()(*args)
