import math
import triton
import triton.language as tl


# Convolution kernel: 3x3 stride=2 padding=1
# Input: X (B, C_in, H, W), weight (C_out, C_in, 3, 3), bias (C_out)
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W, C_out,
    H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    b = tl.program_id(0)  # batch index
    t_out = tl.program_id(1)  # linear index over H_out * W_out
    co_block = tl.program_id(2)  # tile over C_out

    h_out_idx = t_out // W_out
    w_out_idx = t_out % W_out

    co_start = co_block * BLOCK_CO
    co = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co < C_out

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                in_h = h_out_idx * 2 + kh
                in_w = w_out_idx * 2 + kw

                # Load input X[b, ci, in_h, in_w]
                x_ptrs = X_ptr + b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=True, other=0.0).to(tl.float32)

                # Load weights W[co, ci, kh, kw] for this output channel tile: vector over co
                w_base = W_ptr + co * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_base, mask=co_mask, other=0.0).to(tl.float32)

                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS_ptr + co
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store output Y[b, co, h_out_idx, w_out_idx]
    y_ptrs = Y_ptr + b * stride_yb + co * stride_yc + h_out_idx * stride_yh + w_out_idx * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# GELU via tanh approximation (to match PyTorch F.gelu default)
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


# Linear projection (no bias): Y = X @ W^T
# X: (B, T, K) with strides, W: (M, K) with strides (M=1024, K=3840), output Y: (B, T, M)
@triton.jit
def linear_no_bias_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # tile over output channels
    BLOCK_K: tl.constexpr,  # reduce chunk over K
):
    b = tl.program_id(0)  # batch index
    m_block = tl.program_id(1)  # tile over output channels M
    t = tl.program_id(2)  # time index

    m_start = m_block * BLOCK_M
    m = m_start + tl.arange(0, BLOCK_M)
    m_mask = m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K

        # Load X[b, t, k] as vector
        x_ptrs = X_ptr + b * stride_xb + t * stride_xt + k * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[m, k] as (BLOCK_M, BLOCK_K)
        w_ptrs = W_ptr + m[:, None] * stride_wm + k[None, :] * stride_wk
        w_chunk = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate dot products per m
        acc += tl.sum(w_chunk * x_vec[None, :], axis=1)

    # Store result Y[b, t, m]
    y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + m * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Scale + add positional embedding: Y = X + scale * PE
@triton.jit
def add_scaled_pos_emb_kernel(X_ptr, PE_ptr, Y_ptr, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pe = tl.load(PE_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x + scale * pe
    tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, time_dim: int, device: torch.device):
        super().__init__()
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.device = device
        self.embed_scale = math.sqrt(1024)  # d_model = 1024

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
        device = self.device
        dtype = torch.bfloat16  # match get_inputs
        B = self.batch_size
        T = self.time_dim

        # Stage 1: Conv1 (1 -> 384) + GELU
        C_in1 = 1
        C_out1 = 384
        H1 = 80
        W1 = T
        H_out1 = (H1 - 3) // 2 + 1  # 40
        W_out1 = (W1 - 3) // 2 + 1  # T//2

        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=dtype, device=device)

        stride_xb1, stride_xc1, stride_xh1, stride_xw1 = input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3)
        stride_wo1, stride_wi1, stride_wkh1, stride_wkw1 = conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3)
        stride_yb1, stride_yc1, stride_yh1, stride_yw1 = y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3)

        # Launch conv1
        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            stride_xb1, stride_xc1, stride_xh1, stride_xw1,
            stride_wo1, stride_wi1, stride_wkh1, stride_wkw1,
            stride_yb1, stride_yc1, stride_yh1, stride_yw1,
            BLOCK_CO=64,
        )

        # GELU on y1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384) + GELU
        C_in2 = C_out1
        C_out2 = C_out1
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  #


def run(*args):
    return ModelNew()(*args)
