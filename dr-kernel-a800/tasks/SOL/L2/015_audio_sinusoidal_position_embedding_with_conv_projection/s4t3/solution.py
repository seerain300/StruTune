import math
import triton
import triton.language as tl


# Triton conv2d: 3x3, stride=2, padding=1
# Input X: (B, C_in, H, W), weight W: (C_out, C_in, 3, 3), bias BIAS: (C_out)
# Output Y: (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    HW_out = H_out * W_out
    h_out = pid_hw // W_out
    w_out = pid_hw % W_out

    co_start = tl.program_id(2) * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # initialize accumulator
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # loop over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                in_h = h_out * 2 + kh
                in_w = w_out * 2 + kw
                in_h_ok = (in_h >= 0) & (in_h < H)
                in_w_ok = (in_w >= 0) & (in_w < W)
                in_valid = in_h_ok & in_w_ok
                x_ptrs = X_ptr + b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)
                w_base = W_ptr + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_base, mask=co_mask, other=0.0).to(tl.float32)
                acc += x_val * w_vec

    # add bias
    bias_ptrs = BIAS_ptr + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # store result
    y_ptrs = Y_ptr + b * stride_yb + co_offsets * stride_yc + h_out * stride_yh + w_out * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# Triton GELU (tanh approximation): Y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y_ptr + offsets, gelu, mask=mask)


# Triton Linear projection (no bias): Y = X @ W^T
# X: (B, T, K), W: (M, K), Y: (B, T, M)
# We tile over output channels M and reduce over K in chunks.
@triton.jit
def linear_no_bias_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    m_start = tl.program_id(2) * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load X[b, t, k_offsets] -> (BLOCK_K,)
        x_ptrs = X_ptr + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # load W[m_offsets, k_offsets] -> (BLOCK_M, BLOCK_K)
        w_ptrs = W_ptr + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # store Y[b, t, m_offsets]
    y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton elementwise: add scaled positional embedding
# Y: (B, T, D), POS: (S, D) where S >= T, SCALE: float
# We add scaled POS[t, :] to Y[b, t, :] across D.
@triton.jit
def add_scaled_pos_emb_kernel(
    Y_ptr, POS_ptr, SCALE, B, T, D,
    stride_yb, stride_yt, stride_yd,
    stride_pos_s, stride_pos_d,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d_start = tl.program_id(2) * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + d_offsets * stride_yd
    y_vals = tl.load(y_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    pos_ptrs = POS_ptr + t * stride_pos_s + d_offsets * stride_pos_d
    pos_vals = tl.load(pos_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    y_vals += SCALE * pos_vals

    tl.store(y_ptrs, y_vals, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor,
        conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor,
        conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor,
        conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,  # (1024, 3840)
        positional_embedding: torch.Tensor,  # (1500, 1024)
        embed_scale: float,
    ):
        # Ensure device and dtype
        device = input_features.device
        dtype = torch.bfloat16
        input_features = input_features.to(device=device, dtype=dtype).contiguous()
        conv2d1_weight = conv2d1_weight.to(device=device, dtype=dtype).contiguous()
        conv2d1_bias = conv2d1_bias.to(device=device, dtype=dtype).contiguous()
        conv2d2_weight = conv2d2_weight.to(device=device, dtype=dtype).contiguous()
        conv2d2_bias = conv2d2_bias.to(device=device, dtype=dtype).contiguous()
        conv2d3_weight = conv2d3_weight.to(device=device, dtype=dtype).contiguous()
        conv2d3_bias = conv2d3_bias.to(device=device, dtype=dtype).contiguous()
        conv_out_weight = conv_out_weight.to(device=device, dtype=dtype).contiguous()
        positional_embedding = positional_embedding.to(device=device, dtype=dtype).contiguous()

        B = input_features.shape[0]
        T = input_features.shape[3]

        # Stage 1: Conv1 (1 -> 384)
        C_in1 = 1
        C_out1 = 384
        H1 = 80
        W1 = T
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T // 2

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)

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

        # Stage 2: Conv2 (384 -> 384)
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T // 4

        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=dtype, device=device)

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

        # Stage 3: Conv3 (


def run(*args):
    return ModelNew()(*args)
