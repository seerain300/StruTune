import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv3x3_stride2_nchw_kernel(
    x_ptr,          # *bf16 or *f16, input [B, C_in, H, W]
    w_ptr,          # *bf16 or *f16, weight [C_out, C_in, 3, 3]
    bsz,            # int, batch size
    Cin,            # int, input channels
    H, W,           # int, input height/width
    Cout,           # int, output channels
    H_out, W_out,   # int, output height/width
    bias_ptr,       # *bf16, bias [Cout]
    # dtype is inferred from loaded tensors; compute in fp32
):
    b = tl.program_id(0)  # grid dim 0 over batch
    co = tl.program_id(1) # grid dim 1 over output channels

    # Prepare output vector y for all spatial positions (length H_out*W_out)
    pos = 0
    y_vec = tl.zeros((H_out * W_out,), dtype=tl.float32)

    # Accumulate convolution results for each output position
    for oh in range(0, H_out):
        for ow in range(0, W_out):
            acc = 0.0
            # Loop over 3x3 neighborhood with padding=1 (implicit via valid checks)
            for kh in range(0, 3):
                ih = oh + kh - 1  # -1 for padding=1
                valid_h = (ih >= 0) & (ih < H)
                for kw in range(0, 3):
                    iw = ow + kw - 1  # -1 for padding=1
                    valid_w = (iw >= 0) & (iw < W)
                    # If padding would place outside, contribution is zero
                    if valid_h and valid_w:
                        # Loop over input channels
                        for ic in range(0, Cin):
                            x_off = ((b * Cin + ic) * H + ih) * W + iw
                            x_val = tl.load(x_ptr + x_off)  # *bf16 or *f16
                            x_val = x_val.to(tl.float32)
                            # Load weight scalar w[co, ic, kh, kw]
                            w_off = co * (Cin * 9) + ic * 9 + kh * 3 + kw
                            w_val = tl.load(w_ptr + w_off).to(tl.float32)
                            acc += x_val * w_val
            # Add bias
            if co < Cout:
                b_val = tl.load(bias_ptr + co).to(tl.float32)
            else:
                b_val = 0.0
            acc += b_val

            # Apply GELU tanh approximation
            c = 0.7978845608028654  # sqrt(2/pi)
            x3 = acc * acc * acc
            tanh_arg = c * (acc + 0.044715 * x3)
            tanh_val = tl.math.tanh(tanh_arg)
            acc = 0.5 * acc * (1.0 + tanh_val)

            y_vec[pos] = acc
            pos += 1

    # Store to out[b, co, :, :] linearized as [B, Cout, H_out, W_out]
    # out pointer layout: out[b, co, oh, ow]
    base_out = (b * Cout + co) * (H_out * W_out)
    for pos in range(0, H_out * W_out):
        tl.store(out_ptr + base_out + pos, y_vec[pos])


@triton.jit
def linear_gemv_kernel(
    x_ptr,   # *bf16 or *f16, input [B, T, K], contiguous
    w_ptr,   # *bf16 or *f16, weight [N, K], contiguous
    y_ptr,   # *bf16 or *f16, output [B, T, N], contiguous
    B: tl.constexpr,  # int
    T: tl.constexpr,  # int
    K: tl.constexpr,  # int
    N: tl.constexpr,  # int
    BLOCK_K: tl.constexpr,  # tile size for K
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)  # output feature index

    acc = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load x[b, t, k_idx]
        x_off = (b * T + t) * K + k_idx
        x_vec = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_vec = x_vec.to(tl.float32)

        # Load w[d, k_idx]
        w_off = d * K + k_idx
        w_vec = tl.load(w_ptr + w_off, mask=k_mask, other=0.0)
        w_vec = w_vec.to(tl.float32)

        # Accumulate dot product for this tile
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Store y[b, t, d]
    y_off = (b * T + t) * N + d
    tl.store(y_ptr + y_off, acc)


@triton.jit
def add_pos_embed_kernel(
    y_ptr,        # *bf16 or *f16, input/output [B, T, N], contiguous
    pos_ptr,      # *bf16 or *f16, positional embedding [T, N], contiguous
    B: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    # Load current y[b, t, d]
    y_off = (b * T + t) * N + d
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    # Load pos_embed[t, d]
    pos_off = t * N + d
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    # Add
    y_val += pos_val

    # Store back
    tl.store(y_ptr + y_off, y_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Ensure Triton kernels are actually launched from forward.
        # Inputs:
        #   input_features: [B, 1, 80, time_dim], bfloat16
        #   conv2d1_weight: [Cout=384, Cin=1, 3, 3], bfloat16
        #   conv2d1_bias: [384], bfloat16
        #   conv2d2_weight: [384, 384, 3, 3], bfloat16
        #   conv2d2_bias: [384], bfloat16
        #   conv2d3_weight: [384, 384, 3, 3], bfloat16
        #   conv2d3_bias: [384], bfloat16
        #   conv_out_weight: [d_model=1024, conv_out_dim=3840], bfloat16
        #   positional_embedding: [max_source_positions, d_model], bfloat16
        #   embed_scale: float

        # Extract arguments
        # Note: In a real harness, args are provided in order. We rely on the order.
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]
        positional_embedding = args[8]  # [T_max, d_model], bfloat16
        embed_scale = args[9]  # float

        device = input_features.device
        dtype = input_features.dtype

        # Ensure all tensors are on GPU and contiguous
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        positional_embedding = positional_embedding.contiguous()

        B, Cin, H, W = input_features.shape
        Cout1 = conv2d1_weight.shape[0]  # 384
        Cin1 = conv2d1_weight.shape[1]   # 1
        K = 3

        # First conv: x -> [B, 384, H_out1, W_out1]
        H_out1 = (H + 2 * 1 - K) // 2 + 1  # padding=1, stride=2
        W_out1 = (W + 2 * 1 - K) // 2 + 1

        x = torch.empty((B, Cout1, H_out1, W_out1), device=device, dtype=torch.float32)  # compute in fp32
        out1 = torch.empty_like(x, dtype=torch.float32)
        # Launch conv kernel for conv1
        grid1 = (B, Cout1)
        conv3x3_stride2_nchw_kernel[grid1](
            input_features, conv2d1_weight, B, Cin, H, W, Cout1, H_out1, W_out1, conv2d1_bias, out1,
        )

        # Apply GELU already in-kernel

        # Second conv: out1 -> [B, 384, H_out2, W_out2]
        Cout2 = conv2d2_weight.shape[0]  # 384
        Cin2 = conv2d2_weight.shape[1]   # 384

        H_out2 = (H_out1 + 2 * 1 - K) // 2 + 1
        W_out2 = (W_out1 + 2 * 1 - K) // 2 + 1

        x2 = torch.empty((B, Cout2, H_out2, W_out2), device=device, dtype=torch.float32)
        out2 = torch.empty_like(x2, dtype=torch.float32)
        grid2 = (B, Cout2)
        conv3x3_stride2_nchw_kernel[grid2](
            out1, conv2d2_weight, B, Cin2, H_out1, W_out1, Cout2, H_out2, W_out2, conv2d2_bias, out2,
        )

        # Third conv: out2 -> [B, 384, H_out3, W_out3]
        Cout3 = conv2d3_weight.shape[0]  # 384
        Cin3 = conv2d3_weight.shape[1]   # 384

        H_out3 = (H_out2 + 2 * 1 - K) // 2 + 1
        W_out3 = (W_out2 + 2 * 1 - K) // 2 + 1

        x3 = torch.empty((B, Cout3, H_out3, W_out3), device=device, dtype=torch.float32)
        out3 = torch.empty_like(x3, dtype=torch.float32)
        grid3 = (B, Cout3)
        conv3x3_stride2_nchw_kernel[grid3](
            out2, conv2d3_weight, B, Cin3, H_out2, W_out2, Cout3, H_out3, W_out3, conv2d3_bias, out3,
        )

        # Reshape: (B, 384, H_out3, W_out3) -> (B, W_out3, 384*H_out3)
        B3, Cout3, H_out3, W_out3 = out3.shape
        K_features = Cout3 * H_out3 * W_out3  # total features after conv3
        x_reshaped = out3.permute(0, 3, 1, 2).contiguous().view(B3, W_out3, K_features)

        # Linear projection to d_model = 1024
        N = 1024
        y_bf16 = torch.empty((B3, W_out3, N), device=device, dtype=dtype)
        y = torch.empty_like(y_bf16, dtype=torch.float32)

        # Launch linear GEMV kernel: compute y[b, t, d] for all b, t, d
        # Note: conv_out_weight is [N, K_features]
        grid_linear = (B3, W_out3, N)
        linear_gemv_kernel[grid_linear](
            x_reshaped, conv_out_weight, y,
            B3, W_out3, K_features, N, BLOCK_K=128,
        )

        # Scale by embed_scale
        y = y * embed_scale

        # Add positional embedding: positional_embedding is [T_max, N], we need to slice to T=W_out3
        # Create pos_embed of shape [W_out3, N] by slicing first W_out3 rows
        T_max, N = positional_embedding.shape
        pos_embed = positional_embedding[:W_out3, :].contiguous()  # [T, N], bfloat16
        # Cast pos_embed to fp32 for addition with y
        pos_embed = pos_embed.to(torch.float32)

        # Launch add_pos_embed_kernel: broadcast add pos[b, t, :] to y[b, t, :]
        # We need y as fp32 for addition; add in fp32, then cast back to original dtype if needed.
        y_fp32 = y
        grid_pos = (B3, W_out3, N)
        add_pos_embed_kernel[grid_pos](
            y_fp32, pos_embed, B3, W_out3, N
        )

        # If original dtype is bfloat16, cast back; original input is bfloat16
        if dtype == torch.bfloat16:
            return y_fp32.to(torch.bfloat16)
        else:
            return y_fp32

# End of ModelNew implementation.


def run(*args):
    return ModelNew()(*args)
