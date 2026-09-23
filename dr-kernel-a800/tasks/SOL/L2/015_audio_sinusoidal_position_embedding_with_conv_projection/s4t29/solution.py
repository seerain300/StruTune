import math
import torch
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# X: (B, C_in, H, W), W: (C_out, C_in, 3, 3), BIAS: (C_out)
# Y: (B, C_out, H_out, W_out), where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)  # linearized output H_out*W_out
    pid_co = tl.program_id(2)  # tile over output channels

    # compute output h,w indices
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # initialize accumulator
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # reduction over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = 2 * h_out + 1 - kh
                in_w = 2 * w_out + 1 - kw
                # valid only if in_h, in_w in [0, H-1] and [0, W_in-1]
                in_valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W_in)

                # base pointer for X[pid_b, ci, in_h, in_w]
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                # load W[co, ci, kh, kw] for all co in tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # FMA
                acc += x_val * w_vec

    # add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # store to Y
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
# X: (B, T, K) with strides
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

    # reduce over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load X[b, t, k_offsets]
        x_ptrs = X + pid_b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # load W[m_offsets, k_offsets]
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # acc += sum_k (X[t, k] * W[m, k]) for this tile
        # w_mat: [BLOCK_M, BLOCK_K], x_vec: [BLOCK_K] -> dot to [BLOCK_M]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add scaled positional embedding to Y: Y = Y + scale * pos
# Y: (B, T, M), pos: (T, M) sliced from positional_embedding, scale: float
@triton.jit
def add_scaled_pos_emb_kernel(Y, pos, scale, B, T, M, BLOCK_M: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    y_ptrs = Y + pid_b * (T * M) + t * M + m_offsets
    pos_ptrs = pos + t * M + m_offsets

    y_val = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    pos_val = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    y_val = y_val + scale * pos_val
    tl.store(y_ptrs, y_val, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, *args):
        # args are: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        assert len(args) == 10, "ModelNew expects 10 inputs as per the original run signature."
        (
            input_features,
            conv2d1_weight, conv2d1_bias,
            conv2d2_weight, conv2d2_bias,
            conv2d3_weight, conv2d3_bias,
            conv_out_weight,
            positional_embedding,
            embed_scale,
        ) = args

        # Ensure dtype is bfloat16 for I/O; compute in fp32 internally
        input_features = input_features.to(torch.bfloat16).to(self.device)
        dtype = torch.bfloat16

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, C_in1, H1, W1 = input_features.shape
        C_out1 = 384
        H_out1 = (H1 - 3) // 2 + 1
        W_out1 = (W1 - 3) // 2 + 1

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.bfloat16, device=self.device)

        x_strides = input_features.stride()
        w_strides = conv2d1_weight.stride()
        y_strides = y1.stride()

        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            w_strides[0], w_strides[1], w_strides[2], w_strides[3],
            y_strides[0], y_strides[1], y_strides[2], y_strides[3],
            BLOCK_CO=64,
        )

        # GELU1 via Triton
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384) + GELU
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4

        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.bfloat16, device=self.device)

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

        # GELU2 via Triton
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384) + GELU
        C_in3 = C_out2
        C_out3 = C_out2
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T // 8

        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.bfloat16, device=self.device)

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

        # GELU3 via Triton
        y3_gelu = torch.empty_like(y3)
        N3 = y3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](y3, y3_gelu, N3, BLOCK=1024)

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10)
        b, c, f, t = y3_gelu.shape
        K = c * f
        x_proj = y3_gelu.permute(0, 3, 1, 2).contiguous().view(b, t, K)  # (B, T//8, 3840)

        # Linear projection to 1024 (no bias) using Triton
        M = 1024
        conv_out_weight = conv_out_weight.to(torch.bfloat16).to(self.device)  # (1024, 3840)

        Y = torch.empty((b, t, M), dtype=torch.bfloat16, device=self.device)
        x_proj_strides = x_proj.stride()
        w_strides = conv_out_weight.stride()
        y_strides = Y.stride()

        grid_linear = (b, t, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid_linear](
            x_proj, conv_out_weight, Y,
            b, t, 3840, M,
            x_proj_strides[0], x_proj_strides[1], x_proj_strides[2],
            w_strides[0], w_strides[1],
            y_strides[0], y_strides[1], y_strides[2],
            BLOCK_M=128, BLOCK_K=128,
        )

        # Scale embeddings
        # pos is (1500, 1024), but we only need first T rows (t = time_after_conv)
        scale = float(embed_scale)
        pos = positional_embedding[:t, :].to(torch.bfloat16).to(self.device)
        # Launch Triton kernel to add scaled pos
        grid_pos = (b, t, triton.cdiv(M, 128))
        add_scaled_pos_emb_kernel[grid_pos](
            Y, pos, scale, b, t, M, BLOCK_M=128
        )

        return Y


def run(*args):
    return ModelNew()(*args)
