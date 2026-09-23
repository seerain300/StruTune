import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# Input: X (B, C_in, H, W) — contiguous or strided
# Weights: W (C_out, C_in, 3, 3) — contiguous or strided
# Bias: BIAS (C_out) — contiguous
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr
):
    # Program IDs: tile over batch, output positions, and output channels
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)

    # Derive output indices
    hw = pid_hw
    h_out = hw // W_out
    w_out = hw % W_out

    # Tile of output channels
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Initialize accumulator for this tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                in_h = h_out * 2 + kh
                in_w = w_out * 2 + kw
                # Validity of input position
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)
                # Load X[b, ci, in_h, in_w] with mask
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)
                # Load W[co, ci, kh, kw] for this tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)
                # Accumulate
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y
    y_ptrs = Y + pid_b * stride_yb + co_offsets * stride_yc + h_out * stride_yh + w_out * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# Triton kernel: GELU (tanh approximation), elementwise
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
# X: (B, T, K) with strides (we'll pass as flattened pointers per (B, T) row)
# W: (M, K) with strides, output Y: (B, T, M) with strides
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile over output channels (M dimension)
    BLOCK_K: tl.constexpr   # tile over reduction dimension (K dimension)
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)  # tile along M (output channels)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Accumulator for this (b, t) and M tile
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # For each m in the tile, compute dot over K chunk
        # We'll build acc[m] = sum_k X[b, t, k] * W[m, k]
        # Load X[b, t, k_offsets]
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] as [BLOCK_M, BLOCK_K] and reduce over K
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store result to Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale + add positional embedding per (b, t) row
# Y_in: (B, T, D), POS: (S, D) (S >= T), embed_scale: scalar
@triton.jit
def add_scaled_pos_emb_kernel(
    Y_in, POS, Y_out,
    B, T, D,
    embed_scale,
    stride_yb, stride_yt, stride_yd,
    stride_pos_s, stride_pos_d,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t = pid_t
    d_start = 0
    while d_start < D:
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        y_ptrs = Y_in + pid_b * stride_yb + t * stride_yt + d_offsets * stride_yd
        y_vals = tl.load(y_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        pos_ptrs = POS + t * stride_pos_s + d_offsets * stride_pos_d
        pos_vals = tl.load(pos_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        y_scaled = y_vals * embed_scale + pos_vals
        out_ptrs = Y_out + pid_b * stride_yb + t * stride_yt + d_offsets * stride_yd
        tl.store(out_ptrs, y_scaled, mask=d_mask)
        d_start += BLOCK_D


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # All computation via Triton kernels; no torch elementwise ops.
        device = input_features.device
        dtype = torch.bfloat16  # match get_inputs

        B = input_features.shape[0]
        H1 = input_features.shape[2]
        W1 = input_features.shape[3]
        C_in1 = input_features.shape[1]  # 1

        # Stage 1: Conv1 (1 -> 384)
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H1 - 3) // 2 + 1
        W_out1 = (W1 - 3) // 2 + 1

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)

        # Ensure contiguous for predictable strides
        x1 = input_features.contiguous()
        w1 = conv2d1_weight.contiguous()
        b1 = conv2d1_bias.contiguous()
        y1_strides = y1.stride()
        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            x1, w1, b1, y1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1_strides[0], y1_strides[1], y1_strides[2], y1_strides[3],
            BLOCK_CO=64,
        )

        # GELU1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384)
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = y1_gelu.shape[2]
        W2 = y1_gelu.shape[3]
        H_out2 = (H2 - 3) // 2 + 1
        W_out2 = (W2 - 3) // 2 + 1

        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=dtype, device=device)
        x2 = y1_gelu.contiguous()
        w2 = conv2d2_weight.contiguous()
        b2 = conv2d2_bias.contiguous()

        y2_strides = y2.stride()
        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            x2, w2, b2, y2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
            BLOCK_CO=64,
        )

        # GELU2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384)
        C_in3 = C_out2
        C_out3 = C_out2
        H3 = y2_gelu.shape[2]
        W3 = y2_gelu.shape[3]
        H_out3 = (H3 - 3) // 2 + 1
        W_out3 = (W3 - 3) // 2 + 1

        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=dtype, device=device)
        x3 = y2_gelu.contiguous()
        w3 = conv2d3_weight.contiguous()
        b3 = conv2d3_bias.contiguous()

        y3_strides = y3.stride()
        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            x3, w3, b3, y3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
            BLOCK_CO=64,
        )

        # GELU3
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Linear projection: (B, H_out3, W_out3) -> (B, T_after_conv, 1024)
        B3, C_out3, H_out3, W_out3 = y3_gelu.shape
        T_after_conv = W_out3  # per your setup, W_out3 = T // 8 (e.g., 1688 // 8 = 211)
        K = conv_out_weight.shape[1]  # 3840
        M = conv_out_weight.shape[0]  # 1024

        # Reshape X to (B, T_after_conv, K)
        X_for_linear = y3_gelu.view(B3, T_after_conv, K)

        # Ensure contiguous for kernel (though we pass strides)
        X_for_linear = X_for_linear.contiguous()
        W = conv_out_weight.contiguous()
        Y_linear = torch.empty((B3, T_after_conv, M), dtype=dtype, device=device)

        # Strides
        stride_xb, stride_xt, stride_xk = X_for_linear.stride(0), X_for_linear.stride(1), X_for_linear.stride(2)
        stride_wm, stride_wk = W.stride(0), W.stride(1)
        stride_yb, stride_yt, stride_ym = Y_linear.stride(0), Y_linear.stride(1), Y_linear.stride(2)

        # Launch linear kernel
        grid_linear = (B3, T_after_conv, triton.cdiv(M, 64))
        linear_no_bias_kernel[grid_linear](
            X_for_linear, W, Y_linear,
            B3, T_after_conv, K, M,
            stride_xb, stride_xt, stride_xk,
            stride_wm, stride_wk,
            stride_yb, stride_yt, stride_ym,
            BLOCK_M=64, BLOCK_K=256,
        )

        # Scale + add positional embedding (only first T_after_conv rows)
        Y_scaled = torch.empty_like(Y_linear)
        S = positional_embedding.shape[0]  # 1500
        D = positional_embedding.shape[1]  # 1024
        embed_scale = float(embed_scale)  # 32.0

        stride_yb, stride_yt, stride_yd = Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2)
        stride_pos_s, stride_pos_d = positional_embedding.stride(0), positional_embedding.stride(1)

        # We need to slice positional_embedding to (T_after_conv, D) before adding, but Triton expects full tensor.
        # However, we only use first T_after_conv rows in addition, so load with mask on s < T_after_conv.
        # To avoid illegal loads for s >= T_after_conv, we mask loads by s < T_after_conv.
        # But positional_embedding is (S, D) and we only add up to T_after_conv rows; we can simply load with mask on s.
        grid_pos = (B3, T_after_conv)
        # We need a 3D grid for kernel; create a dummy third dim by looping over D tiles
        # Triton grid is static, so we use BLOCK_D = 256 and iterate over D in host by launching with cdiv(D, 256).
        # But Triton kernels require static grid. Instead, we pass a grid for (b, t) and iterate over D tiles in kernel.
        # We can emulate this by setting the third grid dim to 1 and looping over D tiles inside kernel using while.
        # To ensure coverage, we set grid's third dimension to cdiv(D, 256) and kernel will handle tiles via while.
        grid_pos = (B3, T_after_conv, triton.cdiv(D, 256))
        add_scaled_pos_emb_kernel[grid_pos](
            Y_linear, positional_embedding, Y_scaled,
            B3, T_after_conv, D,
            embed_scale,
            stride_yb, stride_yt, stride_yd,
            stride_pos_s, stride_pos_d,
            BLOCK_D=256,
        )

        return Y_scaled


def run(*args):
    return ModelNew()(*args)
