import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# X: (B, C_in, H, W), W: (C_out, C_in, 3, 3), BIAS: (C_out)
# Y: (B, C_out, H_out, W_out), H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    # grid = (B, H_out * W_out, ceil(C_out / BLOCK_CO))
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co_block = tl.program_id(2)

    # Decode h_out and w_out from pid_hw
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    # Tile of output channels
    co_start = pid_co_block * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Iterate over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = h_out * 2 + kh
                in_w = w_out * 2 + kw
                # Load X[b, ci, in_h, in_w]
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                # Bounds mask for input access (always valid due to padding logic)
                in_bounds = True  # padding=1 keeps all positions within bounds
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                # Load W[co, ci, kh, kw] for this output channel tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)
                # Accumulate
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y[b, co, h_out, w_out]
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
# X: (B, T, K), W: (M, K), Y: (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile over output channels M
    BLOCK_K: tl.constexpr,  # tile over reduction K
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

        # Load X[b, t, k_offsets] -> shape (BLOCK_K,)
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m_offsets, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # acc[m] += sum_k (x_vec[k] * w_mat[m, k])
        # Implement as dot: (BLOCK_M,) = (BLOCK_M, BLOCK_K) @ (BLOCK_K,) -> sum over K
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Scale and add positional embedding
# X: (B, T, M), POS: (T, M) (we will pass a slice based on actual T), SCALE: float
@triton.jit
def add_scaled_pos_emb_kernel(X, POS, Y, B, T, M, SCALE, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x_ptrs = X + pid_b * stride_xb + t * stride_xt + m_offsets * stride_xm
    pos_ptrs = POS + t * stride_pt + m_offsets * stride_pm
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym

    x = tl.load(x_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    y = x + pos * SCALE
    tl.store(y_ptrs, y, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,
        conv2d1_weight,
        conv2d1_bias,
        conv2d2_weight,
        conv2d2_bias,
        conv2d3_weight,
        conv2d3_bias,
        conv_out_weight,
        positional_embedding,
        embed_scale,
    ):
        # Ensure contiguity and dtype
        dtype = torch.bfloat16
        device = input_features.device
        input_features = input_features.to(dtype).contiguous()
        conv2d1_weight = conv2d1_weight.to(torch.float32).contiguous()
        conv2d1_bias = conv2d1_bias.to(torch.float32).contiguous()
        conv2d2_weight = conv2d2_weight.to(torch.float32).contiguous()
        conv2d2_bias = conv2d2_bias.to(torch.float32).contiguous()
        conv2d3_weight = conv2d3_weight.to(torch.float32).contiguous()
        conv2d3_bias = conv2d3_bias.to(torch.float32).contiguous()
        conv_out_weight = conv_out_weight.to(torch.float32).contiguous()
        positional_embedding = positional_embedding.to(torch.float32).contiguous()

        B = input_features.shape[0]
        H1 = input_features.shape[2]
        W1 = input_features.shape[3]

        # Stage 1: Conv1 (1 -> 384), stride=2, padding=1
        C_in1 = 1
        C_out1 = 384
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T // 2
        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=device)

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

        # Stage 2: Conv2 (384 -> 384), stride=2, padding=1
        C_in2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4

        y2 = torch.empty((B, C_in2, H_out2, W_out2), dtype=torch.float32, device=device)
        x2 = y1_gelu.contiguous()

        x_strides2 = x2.stride()
        w_strides2 = conv2d2_weight.stride()
        y2_strides = y2.stride()
        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_in2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            x2, conv2d2_weight, conv2d2_bias, y2,
            B, C_in2, H2, W2, C_in2, H_out2, W_out2,
            x_strides2[0], x_strides2[1], x_strides2[2], x_strides2[3],
            w_strides2[0], w_strides2[1], w_strides2[2], w_strides2[3],
            y2_strides[0], y2_strides[1], y2_strides[2], y2_strides[3],
            BLOCK_CO=64,
        )

        # GELU2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384), stride=2, padding=1
        C_in3 = C_in2
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T // 8

        y3 = torch.empty((B, C_in3, H_out3, W_out3), dtype=torch.float32, device=device)
        x3 = y2_gelu.contiguous()

        x_strides3 = x3.stride()
        w_strides3 = conv2d3_weight.stride()
        y3_strides = y3.stride()
        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_in3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            x3, conv2d3_weight, conv2d3_bias, y3,
            B, C_in3, H3, W3, C_in3, H_out3, W_out3,
            x_strides3[0], x_strides3[1], x_strides3[2], x_strides3[3],
            w_strides3[0], w_strides3[1], w_strides3[2], w_strides3[3],
            y3_strides[0], y3_strides[1], y3_strides[2], y3_strides[3],
            BLOCK_CO=64,
        )

        # GELU3
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Reshape to (B, T_after_conv, C_out_dim)
        # T_after_conv = W_out3 * H_out3 = (T//8) * 10
        T_after = W_out3 * H_out3
        y3_gelu_flat = y3_gelu.view(B, T_after, -1).contiguous()  # shape (B, T_after, 384*10) == (B, T_after, 3840)

        # Linear projection: Y = y3_gelu_flat @ conv_out_weight^T (no bias)
        # conv_out_weight: (1024, 3840)
        B2, T2, K = y3_gelu_flat.shape  # B2 == B, T2 == T_after, K == 3840
        M = conv_out_weight.shape[0]    # 1024
        Kdim = conv_out_weight.shape[1] # 3840

        # Output Y (B, T_after, M)
        Y = torch.empty((B, T_after, M), dtype=torch.float32, device=device)

        y_strides = y3_gelu_flat.stride()
        w_strides = conv_out_weight.stride()
        Y_strides = Y.stride()

        # Launch linear kernel
        grid_linear = (B, T_after, triton.cdiv(M, 64))
        linear_no_bias_kernel[grid_linear](
            y3_gelu_flat, conv_out_weight, Y,
            B, T_after, Kdim, M,
            y_strides[0], y_strides[1], y_strides[2],
            w_strides[0], w_strides[1],
            Y_strides[0], Y_strides[1], Y_strides[2],
            BLOCK_M=64, BLOCK_K=128,
        )

        # Scale by embed_scale (float)
        scaled_Y = Y * embed_scale

        # Add positional embedding: only first T_after rows are needed
        # positional_embedding: (1500, 1024) float32
        pos_slice = positional_embedding[:T_after, :].contiguous()  # shape (T_after, 1024)

        # Output tensor for final result
        final = torch.empty_like(scaled_Y)  # (B, T_after, M)

        # Launch add_scaled_pos_emb kernel
        # Note: we assume BLOCK_M = 128 for better performance on M=1024
        grid_pos = (B, T_after, triton.cdiv(M, 128))
        add_scaled_pos_emb_kernel[grid_pos](
            scaled_Y, pos_slice, final,
            B, T_after, M, embed_scale,
            BLOCK_M=128,
            stride_xb=scaled_Y.stride(0), stride_xt=scaled_Y.stride(1), stride_xm=scaled_Y.stride(2),
            stride_pt=pos_slice.stride(0), stride_pm=pos_slice.stride(1),
            stride_yb=final.stride(0), stride_yt=final.stride(1), stride_ym=final.stride(2),
        )

        # Return in bfloat16 (matching get_inputs)
        return final.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
