import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: dense linear Y = X @ W^T + B, where
# X is [B, S, D_in], W is [D_out, D_in], Y is [B, S, D_out].
# We launch one program per (b, s) and loop over D_out and D_in.
@triton.jit
def linear_2d_kernel(
    X_ptr,        # *fp32, input [B, S, D_in]
    W_ptr,        # *fp32, weight [D_out, D_in]
    B_ptr,        # *fp32, bias [D_out] or nullptr
    Y_ptr,        # *fp32, output [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xk,
    stride_w0, stride_w1,
    stride_yb, stride_ys, stride_yk,
    BLOCK_D_OUT: tl.constexpr,
    BLOCK_D_IN: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    # Accumulator for output vector of length D_out
    acc = tl.zeros((BLOCK_D_OUT,), dtype=tl.float32)

    # Loop over output dimensions in tiles
    for d_out_start in range(0, D_out, BLOCK_D_OUT):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_D_OUT)
        # Accumulate over input K dimension in tiles
        for d_in_start in range(0, D_in, BLOCK_D_IN):
            d_in_offsets = d_in_start + tl.arange(0, BLOCK_D_IN)

            # Load X[b, s, d_in_offsets] -> vector [BLOCK_D_IN]
            x = tl.load(
                X_ptr + b * stride_xb + s * stride_xs + d_in_offsets * stride_xk,
                mask=d_in_offsets < D_in,
                other=0.0
            )

            # Load W[d_out_offsets, d_in_offsets] -> matrix [BLOCK_D_OUT, BLOCK_D_IN]
            w = tl.load(
                W_ptr + d_out_offsets[:, None] * stride_w0 + d_in_offsets[None, :] * stride_w1,
                mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in),
                other=0.0
            )

            # acc += sum(x * w, axis=1)
            # x: [BLOCK_D_IN], w: [BLOCK_D_OUT, BLOCK_D_IN]
            # Multiply x by each row of w: broadcast x[None, :] * w, then reduce over columns
            prod = x[None, :] * w  # [BLOCK_D_OUT, BLOCK_D_IN]
            acc += tl.sum(prod, axis=1)

        # Add bias
        if B_ptr != 0:
            bias = tl.load(B_ptr + d_out_offsets, mask=d_out_offsets < D_out, other=0.0)
            acc += bias

        # Store Y[b, s, d_out_offsets]
        tl.store(
            Y_ptr + b * stride_yb + s * stride_ys + d_out_offsets * stride_yk,
            acc,
            mask=d_out_offsets < D_out
        )


# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
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
# Rotation: q1 = x[..., :64], q2 = x[..., 64:], rotated_half = cat((-q2, q1), -1)
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
    # Compute rotation: take first and second half
    q1 = x[0:64]
    q2 = x[64:128]
    rotated_half = tl.concatenate([-q2, q1], axis=0)  # [128]

    # Load cos and sin for this (s, d/2) and broadcast across D
    # Note: cos shape [S, 64], sin shape [S, 64]
    s_idx = s
    half = D // 2  # 64
    cos_vals = tl.load(C_ptr + s_idx * stride_c0 + tl.arange(0, half) * stride_c1, mask=tl.arange(0, half) < half, other=0.0)  # [64]
    sin_vals = tl.load(S_ptr + s_idx * stride_s0 + tl.arange(0, half) * stride_s1, mask=tl.arange(0, half) < half, other=0.0)  # [64]

    # Broadcast cos/sin to full D: first half uses cos_vals, second half uses sin_vals
    # We build full vectors by repeating cos_vals and sin_vals
    cos_full = tl.concatenate([cos_vals, cos_vals + 0.0], axis=0)  # [128]
    sin_full = tl.concatenate([sin_vals, sin_vals + 0.0], axis=0)  # [128]
    y = x * cos_full + rotated_half * sin_full
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # We assume the fixed configurations from the original code:
        self.head_dim = 128
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # hidden_states: [B, S, H*head_dim] where H*head_dim == 12288 (96*128)
        B, S, D_in = hidden_states.shape
        assert D_in == self.num_attention_heads * self.head_dim, "hidden_states last dim must be num_attention_heads * head_dim"

        # Make sure tensors are fp32 and contiguous for Triton
        hidden_states = hidden_states.to(torch.float32).contiguous()
        q_proj_weight = q_proj_weight.to(torch.float32).contiguous()
        k_proj_weight = k_proj_weight.to(torch.float32).contiguous()
        v_proj_weight = v_proj_weight.to(torch.float32).contiguous()
        q_proj_bias = q_proj_bias.to(torch.float32).contiguous() if q_proj_bias is not None else None
        k_proj_bias = k_proj_bias.to(torch.float32).contiguous() if k_proj_bias is not None else None
        v_proj_bias = v_proj_bias.to(torch.float32).contiguous() if v_proj_bias is not None else None
        o_proj_weight = o_proj_weight.to(torch.float32).contiguous()
        q_norm_weight = q_norm_weight.to(torch.float32).contiguous()
        k_norm_weight = k_norm_weight.to(torch.float32).contiguous()
        cos = cos.to(torch.float32).contiguous()  # [S, head_dim/2]
        sin = sin.to(torch.float32).contiguous()  # [S, head_dim/2]

        # Compute Q, K, V via Triton dense linear
        # Output dims: [B, S, head_dim]
        Q = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_q = (B, S)
        linear_2d_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias if q_proj_bias is not None else q_proj_bias, Q,
            B, S, self.head_dim, self.num_attention_heads * self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_D_OUT=self.head_dim, BLOCK_D_IN=64
        )

        K = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_k = (B, S)
        linear_2d_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias if k_proj_bias is not None else k_proj_bias, K,
            B, S, self.head_dim, self.num_attention_heads * self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_D_OUT=self.head_dim, BLOCK_D_IN=64
        )

        V = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_v = (B, S)
        linear_2d_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias if v_proj_bias is not None else v_proj_bias, V,
            B, S, self.head_dim, self.num_attention_heads * self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_D_OUT=self.head_dim, BLOCK_D_IN=64
        )

        # RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        grid_norm_q = (B, self.num_attention_heads, S)
        rmsnorm_kernel[grid_norm_q](
            Q, q_norm_weight, Q_norm,
            B, self.num_attention_heads, S, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),  # Note: we pass 4 strides; kernel expects 3D indexing for [B,H,S,D] and treats H as stride_xh=1. We'll keep correct strides via view/reshape in host.
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            rms_norm_eps
        )

        K_norm = torch.empty_like(K)
        grid_norm_k = (B, self.num_attention_heads, S)
        rmsnorm_kernel[grid_norm_k](
            K, k_norm_weight, K_norm,
            B, self.num_attention_heads, S, self.head_dim,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            rms_norm_eps
        )

        # Rotate (RoPE) Q and K
        Q_rot = torch.empty_like(Q_norm)
        grid_rotate_q = (B, self.num_attention_heads, S)
        # For cos/sin, they are [S, D/2] contiguous; we pass strides accordingly
        rotate_half_kernel[grid_rotate_q](
            Q_norm, cos, sin, Q_rot,
            B, self.num_attention_heads, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
        )

        K_rot = torch.empty_like(K_norm)
        grid_rotate_k = (B, self.num_attention_heads, S)
        rotate_half_kernel[grid_rotate_k](
            K_norm, cos, sin, K_rot,
            B, self.num_attention_heads, S, self.head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
        )

        # Grouped Query Attention: repeat KV heads to match attention heads
        # num_key_value_heads=8, num_key_value_groups=12 => each attention head h uses kv_head = (h // 12) * (8//12)=2 * group, but since num_key_value_heads=8, groups=12, it's actually: num_key_value_groups=12, num_key_value_heads=8, so kv_per_group = 8/12=2/3, not an integer; the original code uses num_key_value_groups=12, num_key_value_heads=8, and num_attention_heads=96.
        # With num_attention_heads divisible by num_key_value_groups (96 % 12 == 0), each attention head h maps to kv_head = (h // num_key_value_groups) * (num_key_value_heads // num_key_value_groups) = h // 12 * 8
        kv_per_group = self.num_key_value_heads // self.num_key_value_groups  # 8 // 12 -> 0.666, but we cannot do this as integer. Instead, rely on the original logic that repeats KV across groups. For simplicity, we compute effective KV by index mapping.
        # However, PyTorch code uses expand + reshape for repetition; we can mimic by using expand without heavy computation:
        # We need to produce [B, num_attention_heads, S, head_dim] from [B, num_key_value_heads, S, head_dim]
        # Since each attention head h uses the same KV head index mapped as above, we can compute KV index per h:
        # kv_index = (h // num_key_value_groups) * kv_per_group, where kv_per_group = num_key_value_heads // num_key_value_groups. In our case, 8 // 12 = 0, which would be problematic. The original code uses expand and reshape. To avoid mistakes, we'll use PyTorch expand for this step (it's a view, no compute).
        K_rot = K_rot[:, :self.num_key_value_heads, :, :].expand(B, self.num_attention_heads, S, self.head_dim).reshape(B, self.num_attention_heads, S, self.head_dim)
        V = V[:, :self.num_key_value_heads, :, :].expand(B, self.num_attention_heads, S, self.head_dim).reshape(B, self.num_attention_heads, S, self.head_dim)

        # Compute attention scores: Q_rot @ K_rot^T * scaling
        # Note: We'll perform attention in PyTorch for correctness: scores = Q_rot @ K_rot.transpose(2,3) * scaling
        # Apply causal mask over SxS
        # Q_rot: [B, H, S, D], K_rot: [B, H, S, D] -> scores [B, H, S, S]
        # We'll compute per (b, h, s) row: attn_weights[b, h, s, :] = softmax_over_keys(scores[b, h, s, :])
        attn_output = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)

        # Compute scores and softmax per (b,h,s) and then output via torch.matmul
        for b in range(B):
            for h in range(self.num_attention_heads):
                # scores [S, S] for this (b, h)
                scores = torch.matmul(Q_rot[b, h], K_rot[b, h].transpose(0, 1)) * self.scaling  # [S, S]
                # causal mask: upper-triangular with diagonal=1
                causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device, dtype=torch.float32), diagonal=1)
                scores = scores + causal_mask
                attn_weights = F.softmax(scores, dim=-1)  # [S, S]
                # output = attn_weights @ V[b, h]  # V is [S, D]
                attn_output[b, h] = torch.matmul(attn_weights, V[b, h])  # [S, D]

        # Transpose and reshape to [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, H*D]
        attn_output = attn_output.reshape(B, S, self.num_attention_heads * self.head_dim)

        # Output projection (no bias): O = attn_output @ o_proj_weight^T
        # Keep in PyTorch for correctness
        O = F.linear(attn_output, o_proj_weight, None)

        return O


def run(*args):
    return ModelNew()(*args)
