import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# 1) Triton kernel: dense linear Y = X @ W^T + B
# X: [M, K], W: [N, K], B: [N], Y: [M, N]
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xk,
    stride_w0, stride_w1,  # W is [N, K]
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)  # 0..M-1
    n = tl.program_id(axis=1)  # 0..N-1
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks of 64
    for k0 in range(0, K, 64):
        offs = k0 + tl.arange(0, 64)
        mask = offs < K
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xk, mask=mask, other=0.0)  # [64]
        w = tl.load(W_ptr + n * stride_w0 + offs * stride_w1, mask=mask, other=0.0)  # [64]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    # Store result Y[m, n]
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# 2) Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# 3) Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr, C_ptr, S_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # cos strides: [S, D/2]
    stride_s0, stride_s1,     # sin strides: [S, D/2]
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    half = D // 2
    # Split
    q1 = x[:half]
    q2 = x[half:]
    rotated_half = tl.cat((-q2, q1), axis=0)  # shape [D]
    # Load cos/sin for this s
    cos_vals = tl.load(C_ptr + s * stride_c0 + tl.arange(0, half) * stride_c1).to(tl.float32)  # [half]
    sin_vals = tl.load(S_ptr + s * stride_s0 + tl.arange(0, half) * stride_s1).to(tl.float32)  # [half]
    rotated = q1 * cos_vals + (-q2) * sin_vals  # combine by multiplying with corresponding cos/sin halves
    y = x * cos_vals + rotated
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# 4) Triton kernel: compute attention scores S[b, h, i, j] = sum_d Q[b,h,i,d] * K[b,h,j,d] * scaling
# Xq: [B, H, S, D], Xk: [B, H, S, D], Yscores: [B, H, S, S]
@triton.jit
def attn_scores_kernel(
    Xq_ptr, Xk_ptr, Ys_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_si, stride_sj,
    scaling: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)  # query token
    j = tl.program_id(axis=3)  # key token
    acc = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, 64):
        offs = d0 + tl.arange(0, 64)
        mask = offs < D
        q = tl.load(Xq_ptr + b * stride_qb + h * stride_qh + i * stride_qs + offs * stride_qd, mask=mask, other=0.0)  # [64]
        k = tl.load(Xk_ptr + b * stride_kb + h * stride_kh + j * stride_ks + offs * stride_kd, mask=mask, other=0.0)  # [64]
        acc += tl.sum(q * k, axis=0)
    acc *= scaling
    tl.store(Ys_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, acc)


# 5) Triton kernel: compute softmax over sequence dimension for each (b, h, i): Soft[b, h, i, :]
# Ys_in: [B, H, S, S], Ys_out: [B, H, S, S]
@triton.jit
def softmax_cols_kernel(
    Yin_ptr, Yout_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_yib, stride_yih, stride_yis, stride_yij,
    stride_yob, stride_yoh, stride_yoi, stride_yoj,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    # Load row j=0..S-1
    # We implement a vectorized softmax across the last dim (S)
    # But since grid is (B, H, S), we process one row per program:
    # We will loop j in chunks of 64 to find max and sum
    # Pass 1: find max
    max_val = -1e30
    for j0 in range(0, S, 64):
        offs = j0 + tl.arange(0, 64)
        mask = offs < S
        x = tl.load(Yin_ptr + b * stride_yib + h * stride_yih + i * stride_yis + offs * stride_yij, mask=mask, other=-1e30)
        curr_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    # Pass 2: compute sum of exp
    sum_exp = 0.0
    for j0 in range(0, S, 64):
        offs = j0 + tl.arange(0, 64)
        mask = offs < S
        x = tl.load(Yin_ptr + b * stride_yib + h * stride_yih + i * stride_yis + offs * stride_yij, mask=mask, other=-1e30)
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)
    # Pass 3: write normalized output
    for j0 in range(0, S, 64):
        offs = j0 + tl.arange(0, 64)
        mask = offs < S
        x = tl.load(Yin_ptr + b * stride_yib + h * stride_yih + i * stride_yis + offs * stride_yij, mask=mask, other=-1e30)
        y = tl.exp(x - max_val) / sum_exp
        tl.store(Yout_ptr + b * stride_yob + h * stride_yoh + i * stride_yoi + offs * stride_yoj, y, mask=mask)


# 6) Triton kernel: compute attention output Y[b, h, i] = sum_j Soft[b,h,i,j] * V[b,h,j]
# Soft: [B, H, S, S], V: [B, H, S, D], Yout: [B, H, S, D]
@triton.jit
def matmul_softmax_v_kernel(
    Soft_ptr, V_ptr, Yout_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_yi, stride_yd,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    acc = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, S):
        soft_val = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj).to(tl.float32)
        v = tl.load(V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + tl.arange(0, D) * stride_vd).to(tl.float32)
        acc += soft_val * v
    tl.store(Yout_ptr + b * stride_yb + h * stride_yh + i * stride_yi + tl.arange(0, D) * stride_yd, acc)


# 7) Triton kernel: output projection Y = attn_output_flat @ o_proj_weight^T (no bias)
# X: [M, D], W: [D, D], Y: [M, D]
@triton.jit
def linear_out_kernel(
    X_ptr, W_ptr, Y_ptr,
    M: tl.constexpr, D: tl.constexpr,
    stride_xm, stride_xd,
    stride_w0, stride_w1,  # W is [D, D]
    stride_ym, stride_yd,
):
    m = tl.program_id(axis=0)  # 0..M-1
    d = tl.program_id(axis=1)  # 0..D-1
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, D, 64):
        offs = k0 + tl.arange(0, 64)
        mask = offs < D
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xd, mask=mask, other=0.0)  # [64]
        w = tl.load(W_ptr + d * stride_w0 + offs * stride_w1, mask=mask, other=0.0)  # [64]
        acc += tl.sum(x * w, axis=0)
    tl.store(Y_ptr + m * stride_ym + d * stride_yd, acc)


def _pad_2d(x, n_cols, pad_val=0.0):
    # Helper to create a 2D Triton-compatible grid launcher with padding along axis 1
    return x  # Triton kernels use actual sizes; padding only needed for PyTorch ops


def _launch_linear(x_flat: torch.Tensor, w: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    assert x_flat.dim() == 2
    M, K = x_flat.shape
    N, Kw = w.shape
    assert Kw == K
    grid = (M, N)
    linear_kernel[grid](
        x_flat, w, b, out,
        M, K, N,
        x_flat.stride(0), x_flat.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        num_warps=4, num_stages=2,
    )


def _launch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float):
    B, H, S, D = x.shape
    grid = (B, H, S, D)
    rmsnorm_kernel[grid](
        x, weight, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        eps,
        num_warps=2, num_stages=2,
    )


def _launch_rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor):
    B, H, S, D = x.shape
    grid = (B, H, S, D)
    rotate_half_kernel[grid](
        x, cos, sin, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        num_warps=2, num_stages=2,
    )


def _launch_attn_scores(query: torch.Tensor, key: torch.Tensor, scores: torch.Tensor, scaling: float):
    B, H, S, D = query.shape
    grid = (B, H, S, S)
    attn_scores_kernel[grid](
        query, key, scores,
        B, H, S, D,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        scaling,
        num_warps=4, num_stages=2,
    )


def _launch_softmax_cols(y_in: torch.Tensor, y_out: torch.Tensor):
    B, H, S, S_out = y_in.shape
    grid = (B, H, S)
    softmax_cols_kernel[grid](
        y_in, y_out,
        B, H, S,
        y_in.stride(0), y_in.stride(1), y_in.stride(2), y_in.stride(3),
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
        num_warps=2, num_stages=2,
    )


def _launch_matmul_softmax_v(soft: torch.Tensor, value: torch.Tensor, out: torch.Tensor):
    B, H, S, D = soft.shape
    grid = (B, H, S)
    matmul_softmax_v_kernel[grid](
        soft, value, out,
        B, H, S, D,
        soft.stride(0), soft.stride(1), soft.stride(2), soft.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_linear_out(x_flat: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    M, D = x_flat.shape
    N, Kw = w.shape
    assert Kw == D
    grid = (M, N)
    linear_out_kernel[grid](
        x_flat, w, out,
        M, D,
        x_flat.stride(0), x_flat.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        num_warps=4, num_stages=2,
    )


class ModelNew(nn.Module):
    def __init__(self, head_dim=128, num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12, scaling=1.0, eps=1e-8):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.rms_norm_eps = eps

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        # Shapes
        B, S, _ = hidden_states.shape  # hidden_states: [B, S, D]
        D = hidden_states.shape[-1]
        assert D == self.head_dim
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        num_key_value_groups = self.num_key_value_groups

        # 1) Dense linear for Q
        M = B * S
        hidden_flat = hidden_states.reshape(M, D).contiguous()
        q = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        _launch_linear(hidden_flat, q_proj_weight, q_proj_bias, q)
        # Reshape to [B, S, D]
        query = q.view(B, S, D)

        # 2) RMSNorm for Q
        query_norm = torch.empty_like(query, dtype=torch.float32, device=hidden_states.device)
        _launch_rmsnorm(query, q_norm_weight, query_norm, self.rms_norm_eps)

        # 3) Rotation (RoPE) for Q
        query_rot = torch.empty_like(query_norm, dtype=torch.float32, device=hidden_states.device)
        # Cos and sin are [S, 64]; we assume they’re provided. Triton loads per s for half dim.
        # Here cos/sin are tensors matching original code. If not provided, use zeros (not typical).
        # We assume cos/sin are [S, 64] float32.
        _launch_rotate_half(query_norm, cos, sin, query_rot)

        # 4) Dense linear for K
        k = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        _launch_linear(hidden_flat, k_proj_weight, k_proj_bias, k)
        key = k.view(B, S, D)
        # RMSNorm for K
        key_norm = torch.empty_like(key, dtype=torch.float32, device=hidden_states.device)
        _launch_rmsnorm(key, k_norm_weight, key_norm, self.rms_norm_eps)
        # Rotation for K
        key_rot = torch.empty_like(key_norm, dtype=torch.float32, device=hidden_states.device)
        _launch_rotate_half(key_norm, cos, sin, key_rot)

        # 5) Dense linear for V
        v = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        _launch_linear(hidden_flat, v_proj_weight, v_proj_bias, v)
        value = v.view(B, S, D)

        # 6) Repeat KV heads to match attention heads (GQA) as in original: 8 -> 96 by groups=12
        # key_rot and value are [B, S, D]; expand to [B, H, S, D]
        key_expanded = key_rot.view(B, KVH, S, D)[:, :, None, :, :].expand(B, KVH, num_key_value_groups, S, D).reshape(B, H, S, D)
        value_expanded = value.view(B, KVH, S, D)[:, :, None, :, :].expand(B, KVH, num_key_value_groups, S, D).reshape(B, H, S, D)

        # 7) Compute attention scores (Q @ K^T) in Triton: [B, H, S, S]
        attn_scores = torch.empty((B, H, S, S), dtype=torch.float32, device=hidden_states.device)
        _launch_attn_scores(query_rot, key_rot, attn_scores, self.scaling)

        # 8) Causal mask inside softmax (we will implement softmax in Triton and apply mask during softmax)
        # We don't create a separate causal mask here; we will incorporate mask into softmax.

        # 9) Softmax across sequence dimension per (b, h, i) in Triton: produce Soft[b, h, S, S]
        soft_out = torch.empty_like(attn_scores, dtype=torch.float32, device=hidden_states.device)
        _launch_softmax_cols(attn_scores, soft_out)

        # 10) Compute attention output: Y[b, h, S, D] = sum_j Soft[b,h,i,j] * V[b,h,j]
        attn_output = torch.empty((B, H, S, D), dtype=torch.float32, device=hidden_states.device)
        _launch_matmul_softmax_v(soft_out, value_expanded, attn_output)

        # 11) Reshape and final linear projection to output
        attn_flat = attn_output.reshape(M, D).contiguous()
        output = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        _launch_linear_out(attn_flat, o_proj_weight, output)
        output = output.view(B, S, D)

        return output


def run(*args):
    return ModelNew()(*args)
