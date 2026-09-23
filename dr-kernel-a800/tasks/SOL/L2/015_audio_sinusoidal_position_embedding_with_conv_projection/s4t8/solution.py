import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# Input: X (B, C_in, H, W) — strided
# Weights: W (C_out, C_in, 3, 3) — strided
# Bias: BIAS (C_out) — contiguous
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    # Program ids: batch, output spatial, output-channel tile
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)

    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Map pid_hw to (h_out, w_out)
    H_outW_out = H_out * W_out
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                in_h = h_out * 2 + kh
                in_w = w_out * 2 + kw
                # Bounds check for input indices
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)
                # Load input X[b, ci, in_h, in_w]
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)
                # Load weights W[co, ci, kh, kw] for this output channel tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y
    y_ptrs = Y + pid_b * stride_yb + co_offsets * stride_yc + h_out * stride_yh + w_out * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# Triton kernel: GELU (tanh approximation), elementwise over N elements
@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, gelu, mask=mask)


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K) with strides
# W: (M, K) with strides
# Y: (B, T, M) with strides
# We will implement a simplified version for demonstration; in practice, torch.linear is preferred.
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile over M (output channels)
    BLOCK_K: tl.constexpr   # tile over K (reduction)
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # shape (BLOCK_K,)
        x_vec = x_vec[:, None]  # (BLOCK_K, 1)

        # Load W[m_offsets, k_offsets]
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        # Accumulate: acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec, axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: scale + add positional embedding (elementwise)
# X: (B, T_after_conv, 1024)
# POS: (seq_len, 1024) with seq_len >= T_after_conv
# SCALE: float32
@triton.jit
def add_scaled_pos_emb_kernel(
    X, POS, Y,
    B, T_after_conv, D,
    stride_xb, stride_xt, stride_xd,
    stride_pos, stride_posd,
    stride_yb, stride_yt, stride_yd,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    t = pid_t
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load X[b, t, d_offsets]
    x_ptrs = X + pid_b * stride_xb + t * stride_xt + d_offsets * stride_xd
    x_vec = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Load POS[0, d_offsets] (since we add for all b, use t as row index in POS; here t <= seq_len)
    pos_ptrs = POS + t * stride_pos + d_offsets * stride_posd
    pos_vec = tl.load(pos_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Scale and add
    y_vec = x_vec * SCALE + pos_vec

    # Store Y[b, t, d_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + d_offsets * stride_yd
    tl.store(y_ptrs, y_vec, mask=d_mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args layout: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]
        positional_embedding = args[8]
        embed_scale = args[9]

        device = input_features.device
        dtype = torch.bfloat16

        # Stage 1: Conv2d (1 -> 384) + GELU
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # GELU in Triton
        x1_gelu = torch.empty_like(x1, dtype=torch.bfloat16, device=device)
        N1 = x1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x1, x1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x2 = F.conv2d(x1_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2_gelu = torch.empty_like(x2, dtype=torch.bfloat16, device=device)
        N2 = x2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x2, x2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x3 = F.conv2d(x2_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3_gelu = torch.empty_like(x3, dtype=torch.bfloat16, device=device)
        N3 = x3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x3, x3_gelu, N3, BLOCK=1024)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        # After conv3, x3_gelu shape: (B, 384, 10, T//8)
        B = x3_gelu.shape[0]
        C = x3_gelu.shape[1]  # 384
        F = x3_gelu.shape[2]  # 10
        T_after_conv = x3_gelu.shape[3]

        x_flat = x3_gelu.permute(0, 3, 1, 2).contiguous().view(B, T_after_conv, C * F)  # (B, T_after_conv, 3840)

        # Linear projection (no bias) using Triton; note: for simplicity and correctness, we use a small M,N
        # Here, we set M=1024 and K=3840 from conv_out_weight
        M = conv_out_weight.shape[0]  # 1024
        K = conv_out_weight.shape[1]  # 3840

        # Allocate output (B, T_after_conv, M)
        y_linear = torch.empty((B, T_after_conv, M), dtype=torch.bfloat16, device=device)

        # Strides
        stride_xb, stride_xt, stride_xk = x_flat.stride(0), x_flat.stride(1), x_flat.stride(2)
        stride_wm, stride_wk = conv_out_weight.stride(0), conv_out_weight.stride(1)
        stride_yb, stride_yt, stride_ym = y_linear.stride(0), y_linear.stride(1), y_linear.stride(2)

        # Launch linear_no_bias_kernel
        BLOCK_M = 64
        BLOCK_K = 128
        grid = (B, T_after_conv, triton.cdiv(M, BLOCK_M))
        linear_no_bias_kernel[grid](
            x_flat, conv_out_weight, y_linear,
            B, T_after_conv, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
        )

        # Scale embeddings
        y_scaled = y_linear.to(torch.float32) * float(embed_scale)  # cast to fp32 for stability
        y_scaled = y_scaled.to(torch.bfloat16)

        # Add positional embedding: positional_embedding shape (1500, 1024), but we only need first T_after_conv rows
        # Create a temporary tensor to hold addition result in bfloat16
        y_final = torch.empty_like(y_scaled)

        # Strides for addition
        stride_xb2, stride_xt2, stride_xd = y_scaled.stride(0), y_scaled.stride(1), y_scaled.stride(2)
        seq_len = positional_embedding.shape[0]
        # Ensure we only use first T_after_conv rows of positional_embedding
        pos_slice = positional_embedding[:T_after_conv, :]  # (T_after_conv, 1024)
        # Strides for pos
        stride_pos = pos_slice.stride(0)
        stride_posd = pos_slice.stride(1)

        stride_yb2, stride_yt2, stride_yd = y_final.stride(0), y_final.stride(1), y_final.stride(2)

        BLOCK_D = 128
        grid_add = (B, T_after_conv, triton.cdiv(1024, BLOCK_D))
        add_scaled_pos_emb_kernel[grid_add](
            y_scaled, pos_slice, y_final,
            B, T_after_conv, 1024,
            stride_xb2, stride_xt2, stride_xd,
            stride_pos, stride_posd,
            stride_yb2, stride_yt2, stride_yd,
            SCALE=embed_scale, BLOCK_D=BLOCK_D
        )

        return y_final


def run(*args):
    return ModelNew()(*args)
