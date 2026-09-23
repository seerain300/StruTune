import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# Input: X (B, C_in, H, W) — strided by passing strides
# Weights: W (C_out, C_in, 3, 3) — strided
# Bias: BIAS (C_out) — strided
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in,  # X's width is passed as W_in
    C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    # Grid: (B, H_out * W_out, C_out_tiles)
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co_tile = tl.program_id(2)

    # Derive h_out and w_out index from pid_hw
    hw_total = H_out * W_out
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    co_start = pid_co_tile * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Accumulator for output channels tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input coordinates with padding=1
                in_h = h_out * 2 + 1 - kh
                in_w = w_out * 2 + 1 - kw
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)

                # Load X[b, ci, in_h, in_w]
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                # Load W[co, ci, kh, kw] for this tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # FMA accumulate
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
# X: (B, T, K) with strides (X is contiguous along T and K; pass strides accordingly)
# W: (M, K) with strides
# Y: (B, T, M) with strides
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

        # Load X[b, t, k_offsets] -> vector of length BLOCK_K
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] -> matrix (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k W[m,k] * X[t,k]
        # w_mat shape (BLOCK_M, BLOCK_K), x_vec shape (BLOCK_K,)
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise multiply by scalar embed_scale
@triton.jit
def mul_scalar_kernel(X, Y, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y + offsets, y, mask=mask)


# Triton kernel: Elementwise add of positional embedding (broadcast across batch)
@triton.jit
def add_pos_emb_kernel(
    X, POS, Y,
    B, T, D,
    stride_xb, stride_xt, stride_xd,
    stride_pos_row, stride_pos_col,
    stride_yb, stride_yt, stride_yd,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    x_ptrs = X + pid_b * stride_xb + pid_t * stride_xt + d_offsets * stride_xd
    x_val = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    pos_ptrs = POS + d_offsets * stride_pos_col  # POS is (1500, 1024), we only use first T rows
    pos_val = tl.load(pos_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    y = x_val + pos_val

    y_ptrs = Y + pid_b * stride_yb + pid_t * stride_yt + d_offsets * stride_yd
    tl.store(y_ptrs, y, mask=d_mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure inputs are contiguous for predictable strides
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        positional_embedding = positional_embedding.contiguous()

        B, C_in1, H1, W1 = input_features.shape
        C_out1 = conv2d1_weight.shape[0]
        # Stage 1: Conv2d (1 -> 384) + GELU
        H_out1 = (H1 - 3) // 2 + 1
        W_out1 = (W1 - 3) // 2 + 1

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=input_features.device)

        x_strides1 = input_features.stride()
        w_strides1 = conv2d1_weight.stride()
        y1_strides = y1.stride()

        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            x_strides1[0], x_strides1[1], x_strides1[2], x_strides1[3],
            w_strides1[0], w_strides1[1], w_strides1[2], w_strides1[3],
            y1_strides[0], y1_strides[1], y1_strides[2], y1_strides[3],
            BLOCK_CO=64,
        )

        # GELU1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384) + GELU
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1
        W_out2 = (W2 - 3) // 2 + 1

        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=input_features.device)

        x_strides2 = y1_gelu.stride()
        w_strides2 = conv2d2_weight.stride()
        y2_strides = y2.stride()

        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            y1_gelu, conv2d2_weight, conv2d2_bias, y2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x_strides2[0], x_strides2[1], x_strides2[2], x_strides2[3],
            w_strides2[0], w_strides2[1], w_strides2[2], w_strides2[3],
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
            BLOCK_CO=64,
        )

        # GELU2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384) + GELU
        C_in3 = C_out2
        C_out3 = C_out2
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1
        W_out3 = (W3 - 3) // 2 + 1

        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=input_features.device)

        x_strides3 = y2_gelu.stride()
        w_strides3 = conv2d3_weight.stride()
        y3_strides = y3.stride()

        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            y2_gelu, conv2d3_weight, conv2d3_bias, y3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x_strides3[0], x_strides3[1], x_strides3[2], x_strides3[3],
            w_strides3[0], w_strides3[1], w_strides3[2], w_strides3[3],
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
            BLOCK_CO=64,
        )

        # GELU3
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Reshape: (B, 384, 10, 1) -> (B, T_after_conv=H_out3*W_out3, C_out_dim=3840)
        B3, C3, H3f, W3f = y3_gelu.shape
        T_after = H3f * W3f  # e.g., 10 * 1 = 10
        y3_gelu_flat = y3_gelu.view(B3, T_after, C3 * H3f * W3f).contiguous()  # (B, T_after, 3840)

        # Linear projection (no bias): Y = X @ W^T -> (B, T_after, 1024)
        B_lin, T_lin, K = y3_gelu_flat.shape  # B_lin=B3, T_lin=T_after, K=3840
        M = conv_out_weight.shape[0]  # 1024

        Y = torch.empty((B_lin, T_lin, M), dtype=torch.float32, device=input_features.device)

        # Strides (assume y3_gelu_flat is contiguous: strides = (T*K, K, 1))
        x_b_stride = T_lin * K
        x_t_stride = K
        x_k_stride = 1

        # W: (M, K), contiguous
        w_m_stride = K
        w_k_stride = 1

        # Y: (B, T, M), contiguous
        y_b_stride = T_lin * M
        y_t_stride = M
        y_m_stride = 1

        grid_lin = (B_lin, T_lin, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid_lin](
            y3_gelu_flat, conv_out_weight, Y,
            B_lin, T_lin, K, M,
            x_b_stride, x_t_stride, x_k_stride,
            w_m_stride, w_k_stride,
            y_b_stride, y_t_stride, y_m_stride,
            BLOCK_M=128, BLOCK_K=128,
        )

        # Multiply by embed_scale
        Y_scaled = torch.empty_like(Y)
        mul_scalar_kernel[(triton.cdiv(Y.numel(), 1024),)](Y, Y_scaled, Y.numel(), embed_scale, BLOCK=1024)

        # Add positional embeddings (only first T_after rows)
        out = torch.empty((B_lin, T_lin, Y_scaled.shape[2]), dtype=torch.float32, device=input_features.device)
        # Strides for add_pos_emb
        x_b_stride_add = T_lin * Y_scaled.shape[2]
        x_t_stride_add = Y_scaled.shape[2]
        x_d_stride_add = 1

        y_b_stride_add = T_lin * Y_scaled.shape[2]
        y_t_stride_add = Y_scaled.shape[2]
        y_d_stride_add = 1

        # POS is (1500, 1024); we only add first T_lin rows
        pos_row_stride = Y_scaled.shape[1]  # 1024
        pos_col_stride = 1

        grid_add = (B_lin, T_lin, triton.cdiv(Y_scaled.shape[2], 128))
        add_pos_emb_kernel[grid_add](
            Y_scaled, positional_embedding, out,
            B_lin, T_lin, Y_scaled.shape[2],
            x_b_stride_add, x_t_stride_add, x_d_stride_add,
            pos_row_stride, pos_col_stride,
            y_b_stride_add, y_t_stride_add, y_d_stride_add,
            BLOCK_D=128,
        )

        return out


def run(*args):
    return ModelNew()(*args)
