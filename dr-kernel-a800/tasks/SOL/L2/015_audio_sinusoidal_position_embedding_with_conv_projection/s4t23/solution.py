import math
import triton
import triton.language as tl


# Triton kernel: Conv2D 3x3, stride=2, padding=1
# Input: X (B, C_in, H, W), Weights: W (C_out, C_in, 3, 3), Bias: BIAS (C_out)
# Output: Y (B, C_out, H_out, W_out) with H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X, W, BIAS, Y,
    B, C_in, H, W_in, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)  # flat over H_out * W_out
    pid_co = tl.program_id(2)  # tile over output channels

    h_out_idx = pid_hw // W_out
    w_out_idx = pid_hw % W_out

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Accumulator for this tile of output channels
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Reduction over input channels and 3x3 taps
    for ci in range(0, C_in):
        for kh in range(0, 3):
            in_h = h_out_idx * 2 + 1 - kh  # padding=1
            if (in_h < 0) or (in_h >= H):
                continue
            for kw in range(0, 3):
                in_w = w_out_idx * 2 + 1 - kw
                if (in_w < 0) or (in_w >= W_in):
                    continue
                # Load X[b, ci, in_h, in_w]
                x_ptrs = X + pid_b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=True, other=0.0).to(tl.float32)

                # Load W[co, ci, kh, kw] for all co in tile
                w_ptrs = W + co_offsets * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # Accumulate
                acc += x_val * w_vec

    # Add bias
    bias_ptrs = BIAS + co_offsets
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y
    y_ptrs = Y + pid_b * stride_yb + co_offsets * stride_yc + h_out_idx * stride_yh + w_out_idx * stride_yw
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
# X: (B, T, K) with strides; here T = time_after_conv, K = 3840
# W: (M, K) with strides; here M = 1024, K = 3840
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

        # Load W[m_offsets, k_offsets]
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # shape (BLOCK_M, BLOCK_K)

        # Accumulate: acc[m] += sum_k x_vec[k] * w_mat[m, k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store Y[b, t, m_offsets]
    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Add scaled positional embedding to Y
# Y: (B, T, M), POS: (T, M) scaled by embed_scale
@triton.jit
def add_scaled_pos_emb_kernel(Y, POS, B, T, M, embed_scale, BLOCK: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    t = pid_t
    m_start = pid_m * BLOCK
    m_offsets = m_start + tl.arange(0, BLOCK)
    m_mask = m_offsets < M

    y_ptrs = Y + pid_b * stride_yb + t * stride_yt + m_offsets * stride_ym
    pos_ptrs = POS + t * stride_pt + m_offsets * stride_pm
    y = tl.load(y_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptrs, mask=m_mask, other=0.0).to(tl.float32) * embed_scale
    y += pos
    tl.store(y_ptrs, y, mask=m_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, time_after_conv: int):
        super().__init__()
        self.time_after_conv = time_after_conv

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor,
        conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor,
        conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor,
        conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,
        positional_embedding: torch.Tensor,
        embed_scale: float,
    ):
        # Ensure dtype is bfloat16 for I/O, but compute in float32 inside kernels
        device = input_features.device
        dtype_io = torch.bfloat16

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, T//2)
        B, C_in1, H, W = input_features.shape
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H - 3) // 2 + 1
        W_out1 = W // 2
        y1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.float32, device=device)  # accumulate in fp32
        input_features_c = input_features.contiguous().to(torch.float32)
        conv2d1_weight_c = conv2d1_weight.contiguous().to(torch.float32)
        conv2d1_bias_c = conv2d1_bias.contiguous().to(torch.float32)
        y1 = y1  # already fp32

        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features_c, conv2d1_weight_c, conv2d1_bias_c, y1,
            B, C_in1, H, W, C_out1, H_out1, W_out1,
            input_features_c.stride(0), input_features_c.stride(1), input_features_c.stride(2), input_features_c.stride(3),
            conv2d1_weight_c.stride(0), conv2d1_weight_c.stride(1), conv2d1_weight_c.stride(2), conv2d1_weight_c.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_CO=64,
        )

        # GELU1
        y1_gelu = torch.empty_like(y1)
        N1 = y1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](y1, y1_gelu, N1, BLOCK=1024)

        # Conv2: (B, 384, 40, T//2) -> (B, 384, 20, T//4)
        C_in2 = C_out1
        C_out2 = C_in2  # 384
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = W2 // 2  # T//4
        y2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=device)
        x2 = y1_gelu.contiguous().to(torch.float32)
        conv2d2_weight_c = conv2d2_weight.contiguous().to(torch.float32)
        conv2d2_bias_c = conv2d2_bias.contiguous().to(torch.float32)

        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            x2, conv2d2_weight_c, conv2d2_bias_c, y2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d2_weight_c.stride(0), conv2d2_weight_c.stride(1), conv2d2_weight_c.stride(2), conv2d2_weight_c.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_CO=64,
        )

        # GELU2
        y2_gelu = torch.empty_like(y2)
        N2 = y2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](y2, y2_gelu, N2, BLOCK=1024)

        # Conv3: (B, 384, 20, T//4) -> (B, 384, 10, T//8)
        C_in3 = C_out2
        C_out3 = C_in3  # 384
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = W3 // 2  # T//8
        y3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=device)
        x3 = y2_gelu.contiguous().to(torch.float32)
        conv2d3_weight_c = conv2d3_weight.contiguous().to(torch.float32)
        conv2d3_bias_c = conv2d3_bias.contiguous().to(torch.float32)

        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            x3, conv2d3_weight_c, conv2d3_bias_c, y3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            conv2d3_weight_c.stride(0), conv2d3_weight_c.stride(1), conv2d3_weight_c.stride(2), conv2d3_weight_c.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_CO=64,
        )

        # Reshape to (B, T_after_conv, K=3840) for linear projection.
        # We are given time_after_conv externally.
        T_after_conv = self.time_after_conv
        # y3 has shape (B, 384, 10, T_after_conv). We need to flatten to (B, T_after_conv, 3840).
        # The original run uses some logic to get T_after_conv implicitly; here we rely on T_after_conv passed in.
        # Compute K from known 384 and 10: K = 384 * (10) = 3840, which matches conv_out_weight shape.
        y3_flat = y3.view(B, H_out3, W_out3).permute(0, 2, 1).contiguous().view(B, T_after_conv, 384 * 10).to(torch.float32)

        # Linear projection: (B, T_after_conv, 3840) @ (1024, 3840)^T -> (B, T_after_conv, 1024)
        M = conv_out_weight.shape[0]  # 1024
        K = conv_out_weight.shape[1]  # 3840
        y = torch.empty((B, T_after_conv, M), dtype=torch.float32, device=device)
        X = y3_flat  # (B, T_after_conv, K)
        W = conv_out_weight.contiguous().to(torch.float32)  # (M, K)

        grid_linear = (B, T_after_conv, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid_linear](
            X, W, y,
            B, T_after_conv, K, M,
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_M=128, BLOCK_K=128,
        )

        # Add scaled positional embedding: y = y + embed_scale * positional_embedding[:T_after_conv, :]
        # positional_embedding is (1500, 1024) in the original setup. We slice first T_after_conv rows.
        pos_scaled = positional_embedding[:T_after_conv, :].contiguous().to(torch.float32) * embed_scale  # (T_after_conv, 1024)
        # Launch Triton kernel to add scaled embedding
        grid_add = (B, T_after_conv, triton.cdiv(M, 1024))
        add_scaled_pos_emb_kernel[grid_add](
            y, pos_scaled, B, T_after_conv, M, embed_scale,
            y.stride(0), y.stride(1), y.stride(2), pos_scaled.stride(0), pos_scaled.stride(1),
            BLOCK=1024,
        )

        # Return y, which is (B, T_after_conv, 1024). Original run uses bfloat16 I/O; here we keep fp32 for stability.
        # If you need bfloat16 output, cast at the end. However, Triton kernels above compute in fp32, so returning fp32 is fine for correctness.
        return y


def run(*args):
    return ModelNew()(*args)
