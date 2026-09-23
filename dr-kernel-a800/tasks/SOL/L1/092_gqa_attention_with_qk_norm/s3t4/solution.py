import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: dense linear Y = X @ W^T + b, where
# X: [B, S, D_in], W: [D_out, D_in], b: [D_out], Y: [B, S, D_out]
# We launch one program per (b, s). Inside, we iterate over D_out and D_in in chunks.
@triton.jit
def linear_kernel(
    X_ptr,        # *fp32, input [B, S, D_in]
    W_ptr,        # *fp32, weight [D_out, D_in]
    BIAS_ptr,     # *fp32, bias [D_out] or None
    Y_ptr,        # *fp32, output [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xd,
    stride_w0, stride_w1,   # W strides: (D_out, D_in)
    stride_yb, stride_ys, stride_yd,
    BLOCK_D_IN: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    # Accumulator for the output vector at (b, s)
    acc = tl.zeros([BLOCK_D_OUT], dtype=tl.float32)
    # Loop over D_out in tiles
    for d_out_start in range(0, D_out, BLOCK_D_OUT):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_D_OUT)
        # Load bias for these d_out_offsets
        bias = tl.zeros([BLOCK_D_OUT], dtype=tl.float32)
        if BIAS_ptr != 0:
            bias = tl.load(BIAS_ptr + d_out_offsets, mask=d_out_offsets < D_out, other=0.0)
        # Iterate over D_in in tiles
        for d_in_start in range(0, D_in, BLOCK_D_IN):
            d_in_offsets = d_in_start + tl.arange(0, BLOCK_D_IN)
            # Load X[b, s, d_in_offsets]
            x = tl.load(
                X_ptr + b * stride_xb + s * stride_xs + d_in_offsets * stride_xd,
                mask=d_in_offsets < D_in,
                other=0.0
            )  # [BLOCK_D_IN]
            # Load W[d_out_offsets, d_in_offsets] as [BLOCK_D_OUT, BLOCK_D_IN]
            w = tl.load(
                W_ptr + d_out_offsets[:, None] * stride_w0 + d_in_offsets[None, :] * stride_w1,
                mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in),
                other=0.0
            )  # [BLOCK_D_OUT, BLOCK_D_IN]
            # Accumulate: acc += sum(x * w, axis=1)
            acc += tl.sum(x[None, :] * w, axis=1)
        # Add bias
        acc += bias
        # Store Y[b, s, d_out_offsets]
        tl.store(
            Y_ptr + b * stride_yb + s * stride_ys + d_out_offsets * stride_yd,
            acc,
            mask=d_out_offsets < D_out
        )

# Triton kernel: RMSNorm per (b, h, s, d)
@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    W_ptr,        # *fp32, weight [D]
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps,          # float32
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )
    x32 = x.to(tl.float32)
    mean_sq = tl.sum(x32 * x32, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d_offsets, mask=d_offsets < D, other=1.0)
    y = (x32 * inv_rms) * w
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )


# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
# Rotation for the first half: q1 = x[..., :64], q2 = x[..., 64:], rotated_half = cat((-q2, q1), -1)
# Final: Y = X * cos + rotated_half * sin
@triton.jit
def rotate_half_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    C_ptr,        # *fp32, cos [S, D/2] flattened
    S_ptr,        # *fp32, sin [S, D/2] flattened
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # strides for cos: (S, D/2)
    stride_s0, stride_s1,     # strides for sin: (S, D/2)
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )  # [D]
    # First half and second half
    q1 = x[:64]
    q2 = x[64:]
    # Load cos and sin for s, d in [0, D//2)
    c = tl.load(
        C_ptr + s * stride_c0 + tl.arange(0, D // 2) * stride_c1,
        mask=tl.arange(0, D // 2) < (D // 2),
        other=1.0
    )
    s_rope = tl.load(
        S_ptr + s * stride_s0 + tl.arange(0, D // 2) * stride_s1,
        mask=tl.arange(0, D // 2) < (D // 2),
        other=0.0
    )
    rotated_half = tl.cat([-q2, q1], axis=0)  # shape [D]
    y = x * c + rotated_half * s_rope
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )


# Triton kernel: attention forward per (b, h). Compute softmax over keys and output.
# We compute attention scores (Q[b,h] @ K[b,h]^T) scaled by 1/sqrt(D), apply causal mask,
# compute softmax per (b,h), and then compute output = softmax @ V[b,h].
# Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D], outputs Soft: [B, H, S, S], Out: [B, H, S, D].
# One program per (b,h), loops over sequence tiles.
@triton.jit
def attention_forward_kernel(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, H, S, D]
    V_ptr,        # *fp32, [B, H, S, D]
    Soft_ptr,     # *fp32, [B, H, S, S] temporary buffer
    Out_ptr,      # *fp32, [B, H, S, D] output
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_sob, stride_soh, stride_sos, stride_sod,
    stride_ob, stride_oh, stride_os, stride_od,
    scale,        # float32
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)

    # Loop over all s positions to produce attention scores and outputs
    for s_i in range(0, S):
        # Initialize softmax vector Soft[b,h,s_i, :] = 0
        for s_j in range(0, S):
            # Compute Q[b,h,s_i,:] @ K[b,h,s_j,:]^T (dot product over D)
            acc = 0.0
            for d_start in range(0, D, 64):
                d_offsets = d_start + tl.arange(0, 64)
                q = tl.load(
                    Q_ptr + b * stride_qb + h * stride_qh + s_i * stride_qs + d_offsets * stride_qd,
                    mask=d_offsets < D,
                    other=0.0
                )  # [64]
                k = tl.load(
                    K_ptr + b * stride_kb + h * stride_kh + s_j * stride_ks + d_offsets * stride_kd,
                    mask=d_offsets < D,
                    other=0.0
                )  # [64]
                acc += tl.sum(q * k, axis=0)
            # Scale and apply causal mask: if s_j <= s_i, keep; else -inf
            causal = 1.0 if s_j <= s_i else 0.0
            acc = acc * scale + (-1e20) * (1.0 - causal)
            tl.store(
                Soft_ptr + b * stride_sob + h * stride_soh + s_i * stride_sos + s_j * stride_sod,
                acc,
                mask=True
            )

        # Compute softmax over Soft[b,h,s_i, :]
        e = tl.zeros([S], dtype=tl.float32)
        for s_j in range(0, S):
            e[s_j] = tl.load(Soft_ptr + b * stride_sob + h * stride_soh + s_i * stride_sos + s_j * stride_sod)
        e = tl.exp(e)
        denom = tl.sum(e, axis=0)
        for s_j in range(0, S):
            soft = e[s_j] / denom
            tl.store(
                Soft_ptr + b * stride_sob + h * stride_soh + s_i * stride_sos + s_j * stride_sod,
                soft
            )

        # Now compute output: Out[b,h,s_i,:] = sum_j Soft[b,h,s_i,j] * V[b,h,s_j,:]
        out_vec = tl.zeros([D], dtype=tl.float32)
        for s_j in range(0, S):
            soft_val = tl.load(Soft_ptr + b * stride_sob + h * stride_soh + s_i * stride_sos + s_j * stride_sod)
            for d_start in range(0, D, 64):
                d_offsets = d_start + tl.arange(0, 64)
                v = tl.load(
                    V_ptr + b * stride_vb + h * stride_vh + s_j * stride_vs + d_offsets * stride_vd,
                    mask=d_offsets < D,
                    other=0.0
                )  # [64]
                out_vec += soft_val * tl.sum(v, axis=0)  # v is [64], sum to scalar
        # Store Out[b,h,s_i, :]
        for d_start in range(0, D, 64):
            d_offsets = d_start + tl.arange(0, 64)
            tl.store(
                Out_ptr + b * stride_ob + h * stride_oh + s_i * stride_os + d_offsets * stride_od,
                out_vec[d_start : d_start + 64],
                mask=d_offsets < D
            )


# Triton kernel: dense linear Y = X @ W^T (no bias), used for O projection: attn_output @ o_proj_weight^T
@triton.jit
def linear_no_bias_kernel(
    X_ptr,        # *fp32, input [B, S, D_in]
    W_ptr,        # *fp32, weight [D_out, D_in]
    Y_ptr,        # *fp32, output [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xd,
    stride_w0, stride_w1,   # W strides: (D_out, D_in)
    stride_yb, stride_ys, stride_yd,
    BLOCK_D_IN: tl.constexpr, BLOCK_D_OUT: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    acc = tl.zeros([BLOCK_D_OUT], dtype=tl.float32)
    for d_out_start in range(0, D_out, BLOCK_D_OUT):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_D_OUT)
        acc = tl.zeros([BLOCK_D_OUT], dtype=tl.float32)
        for d_in_start in range(0, D_in, BLOCK_D_IN):
            d_in_offsets = d_in_start + tl.arange(0, BLOCK_D_IN)
            x = tl.load(
                X_ptr + b * stride_xb + s * stride_xs + d_in_offsets * stride_xd,
                mask=d_in_offsets < D_in,
                other=0.0
            )  # [BLOCK_D_IN]
            w = tl.load(
                W_ptr + d_out_offsets[:, None] * stride_w0 + d_in_offsets[None, :] * stride_w1,
                mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in),
                other=0.0
            )  # [BLOCK_D_OUT, BLOCK_D_IN]
            acc += tl.sum(x[None, :] * w, axis=1)
        tl.store(
            Y_ptr + b * stride_yb + s * stride_ys + d_out_offsets * stride_yd,
            acc,
            mask=d_out_offsets < D_out
        )

# ModelNew: Triton-optimized forward that uses Triton kernels for all heavy compute
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = 12
        assert self.num_attention_heads % self.num_key_value_groups == 0
        assert self.head_dim == 128

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        B, S, D_in = hidden_states.shape
        # Validate shapes
        assert D_in == self.num_attention_heads * self.head_dim, "hidden_states last dim must be num_attention_heads * head_dim"
        # 1) Dense projections: Q, K, V
        Q = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        K = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        V = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        # Launch linear for each (b, s)
        grid_qk = (B, S)
        linear_kernel[grid_qk](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, D_in, self.head_dim, hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_D_IN=64, BLOCK_D_OUT=64
        )
        linear_kernel[grid_qk](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, D_in, self.head_dim, hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_D_IN=64, BLOCK_D_OUT=64
        )
        linear_kernel[grid_qk](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, D_in, self.head_dim, hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_D_IN=64, BLOCK_D_OUT=64
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)
        grid_norm = (B, self.num_attention_heads, S)
        rmsnorm_kernel[grid_norm](
            Q, q_norm_weight, Q_norm, B, self.num_attention_heads, S, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            rms_norm_eps
        )
        rmsnorm_kernel[grid_norm](
            K, k_norm_weight, K_norm, B, self.num_attention_heads, S, self.head_dim,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            rms_norm_eps
        )

        # 3) Rotate half (RoPE) for Q and K
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)
        grid_rot = (B, self.num_attention_heads, S)
        # cos, sin are [S, D/2]; pass strides as (S, D/2)
        rotate_half_kernel[grid_rot](
            Q_norm, cos, sin, Q_rot,
            B, self.num_attention_heads, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1)
        )
        rotate_half_kernel[grid_rot](
            K_norm, cos, sin, K_rot,
            B, self.num_attention_heads, S, self.head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1)
        )

        # 4) Attention forward: compute attention output per (b, h) using Triton
        attn_out = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        soft = torch.empty((B, self.num_attention_heads, S, S), device=hidden_states.device, dtype=torch.float32)
        grid_attention = (B, self.num_attention_heads)
        attention_forward_kernel[grid_attention](
            Q_rot, K_rot, V,
            soft, attn_out,
            B, self.num_attention_heads, S, self.head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            soft.stride(0), soft.stride(1), soft.stride(2), soft.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            1.0 / (self.head_dim ** 0.5)
        )

        # 5) Output projection (O): attn_out @ o_proj_weight^T (no bias)
        output = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_o = (B, S)
        linear_no_bias_kernel[grid_o](
            attn_out, o_proj_weight,
            output,
            B, S, self.head_dim, self.head_dim,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_D_IN=64, BLOCK_D_OUT=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
