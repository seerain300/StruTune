import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, IH, IW, OC, OH, OW,
    # sizes for input/output tensors
    X_B, X_C, X_H, X_W,
    W_OC, W_IC, W_KH, W_KW,
    Y_B, Y_OC, Y_H, Y_W,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
):
    # program ids: grid = (B, OC, OH, OW)
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for ic in range(0, IC):
        for ky in range(0, 3):
            ih = STRIDE * oh + ky - PAD
            in_bounds_h = (ih >= 0) & (ih < IH)
            for kx in range(0, 3):
                iw = STRIDE * ow + kx - PAD
                in_bounds_w = (iw >= 0) & (iw < IW)
                in_bounds = in_bounds_h & in_bounds_w
                # Input pointer: X[b, ic, ih, iw]
                x_ptr = X_ptr + b * X_B + ic * X_C + ih * X_H + iw * X_W
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)
                # Weight pointer: W[oc, ic, ky, kx]
                w_ptr = W_ptr + oc * W_OC + ic * W_IC + ky * W_KH + kx * W_KW
                w_val = tl.load(w_ptr)
                acc += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + oc)
    acc = acc + bias

    # Store output: Y[b, oc, oh, ow]
    y_ptr = Y_ptr + b * Y_B + oc * Y_OC + oh * Y_H + ow * Y_W
    tl.store(y_ptr, acc)


@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, NEL: tl.constexpr):
    # 1D kernel over NEL elements; Y[i] = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    for i in range(0, NEL):
        x = tl.load(X_ptr + i)
        x32 = x.to(tl.float32)
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x32 * x32 * x32
        t = c * (x32 + 0.044715 * x3)
        y = 0.5 * x32 * (1.0 + tl.math.tanh(t))
        tl.store(Y_ptr + i, y)


@triton.jit
def linear_project_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T, N, M,
    scale: tl.float32,
):
    # grid = (B, T, ceil_div(M, 128))
    b = tl.program_id(0)
    t = tl.program_id(1)
    m0 = tl.program_id(2) * 128
    m_offsets = m0 + tl.arange(0, 128)
    mask_m = m_offsets < M
    acc = tl.zeros([128], dtype=tl.float32)

    # Reduce over N in tiles of 256
    for n0 in range(0, N, 256):
        n_offsets = n0 + tl.arange(0, 256)
        mask_n = n_offsets < N
        # Load X[b, t, n_offsets]
        x_ptrs = X_ptr + b * (T * N) + t * N + n_offsets
        x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)  # [256]
        # Load W[m_offsets, n_offsets] -> [128, 256]
        w_ptrs = W_ptr + m_offsets[:, None] * N + n_offsets[None, :]
        w_vals = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        # Accumulate per m
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Apply scale and add pos_emb[t, :]
    acc = acc * scale
    pos_vec = tl.load(pos_ptr + t * M + m_offsets, mask=mask_m, other=0.0)
    acc = acc + pos_vec
    # Store Y[b, t, m_offsets]
    y_ptrs = Y_ptr + b * (T * M) + t * M + m_offsets
    tl.store(y_ptrs, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16 (we will compute in fp32)
        conv2d1_weight: (384, 1, 3, 3), bfloat16
        conv2d1_bias: (384), bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3), bfloat16
        conv2d2_bias, conv2d3_bias: (384), bfloat16
        conv_out_weight: (1024, 3840), bfloat16 (we will compute in fp32)
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float (sqrt(1024) == 32.0)
        """
        B, C, IH, IW = input_features.shape
        device = input_features.device

        # Stage 1: conv1, stride=2, pad=1, 3x3 -> (B, 384, 40, (T+1)//2)
        OC1 = 384
        OH1 = (IH + 1) // 2  # 40
        OW1 = (IW + 1) // 2  # (T + 1)//2
        Y1 = torch.empty((B, OC1, OH1, OW1), dtype=torch.float32, device=device)
        grid1 = (B, OC1, OH1, OW1)
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, Y1,
            B, 1, IH, IW, OC1, OH1, OW1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            Y1.stride(0), Y1.stride(1), Y1.stride(2), Y1.stride(3),
            2, 1,
        )
        # GELU
        Y1_flat = Y1.view(-1)
        Y1_gelu = torch.empty_like(Y1_flat, dtype=torch.float32, device=device)
        gelu_tanh_kernel[(Y1_flat.numel(),)](Y1_flat, Y1_gelu, Y1_flat.numel())
        Y1 = Y1_gelu.view_as(Y1)

        # Stage 2: conv2, stride=2, pad=1, 3x3 -> (B, 384, 20, (T+1)//4)
        OC2 = 384
        OH2 = (OH1 + 1) // 2  # 20
        OW2 = (OW1 + 1) // 2  # (T + 1)//4
        Y2 = torch.empty((B, OC2, OH2, OW2), dtype=torch.float32, device=device)
        grid2 = (B, OC2, OH2, OW2)
        conv2d_stride2_kernel[grid2](
            Y1, conv2d2_weight, conv2d2_bias, Y2,
            B, OC1, OH1, OW1, OC2, OH2, OW2,
            Y1.stride(0), Y1.stride(1), Y1.stride(2), Y1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            Y2.stride(0), Y2.stride(1), Y2.stride(2), Y2.stride(3),
            2, 1,
        )
        # GELU
        Y2_flat = Y2.view(-1)
        Y2_gelu = torch.empty_like(Y2_flat, dtype=torch.float32, device=device)
        gelu_tanh_kernel[(Y2_flat.numel(),)](Y2_flat, Y2_gelu, Y2_flat.numel())
        Y2 = Y2_gelu.view_as(Y2)

        # Stage 3: conv3, stride=2, pad=1, 3x3 -> (B, 384, 10, (T+1)//8)
        OC3 = 384
        OH3 = (OH2 + 1) // 2  # 10
        OW3 = (OW2 + 1) // 2  # (T + 1)//8
        Y3 = torch.empty((B, OC3, OH3, OW3), dtype=torch.float32, device=device)
        grid3 = (B, OC3, OH3, OW3)
        conv2d_stride2_kernel[grid3](
            Y2, conv2d3_weight, conv2d3_bias, Y3,
            B, OC2, OH2, OW2, OC3, OH3, OW3,
            Y2.stride(0), Y2.stride(1), Y2.stride(2), Y2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            Y3.stride(0), Y3.stride(1), Y3.stride(2), Y3.stride(3),
            2, 1,
        )
        # GELU
        Y3_flat = Y3.view(-1)
        Y3_gelu = torch.empty_like(Y3_flat, dtype=torch.float32, device=device)
        gelu_tanh_kernel[(Y3_flat.numel(),)](Y3_flat, Y3_gelu, Y3_flat.numel())
        Y3 = Y3_gelu.view_as(Y3)

        # Final: permute to (B, T_after, 3840), where T_after = OW3 = (T + 1)//8
        T_after = OW3
        Xproj = Y3.permute(0, 3, 1, 2).contiguous().view(B, T_after, OC3 * 10)

        # Linear projection to 1024: Y = Xproj @ W^T, W: (1024, 3840)
        M = conv_out_weight.shape[0]  # 1024
        W = conv_out_weight.contiguous()
        Y = torch.empty((B, T_after, M), dtype=torch.float32, device=device)
        grid_lp = (B, T_after, (M + 127) // 128)
        linear_project_pos_kernel[grid_lp](
            Xproj, W, positional_embedding, Y,
            B, T_after, OC3 * 10, M,
            float(embed_scale),
        )
        return Y


def run(*args):
    return ModelNew()(*args)
