import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv3x3_stride2_nchw_kernel(
    x_ptr,            # *f16/bf16, input: [B, C_in, H, W], NCHW
    w_ptr,            # *f16/bf16, weight: [C_out, C_in, 3, 3]
    bias_ptr,         # *f16/bf16, bias: [C_out]
    out_ptr,          # *f16/bf16, output: [B, C_out, H_out, W_out], NCHW
    B: tl.constexpr,  # batch size (compile-time for grid)
    C_in: tl.constexpr,  # input channels (compile-time for loops; expect 1)
    H: tl.constexpr,  # input height
    W: tl.constexpr,  # input width
    C_out: tl.constexpr,  # output channels
    H_out: tl.constexpr,  # output height
    W_out: tl.constexpr,  # output width
):
    b = tl.program_id(0)
    co = tl.program_id(1)

    # Output vector for this (b, co): length H_out * W_out
    y = tl.zeros((H_out * W_out,), dtype=tl.float32)

    # Accumulate over 3x3 neighborhood and input channels
    for ic in range(0, C_in):
        for ih_out in range(0, H_out):
            for ow_out in range(0, W_out):
                acc = tl.zeros((), dtype=tl.float32)
                # 3x3 neighborhood with zero padding via mask (implicit by range)
                for kh in range(0, 3):
                    ih_k = ih_out + kh - 1  # -1 for padding
                    # if in-bounds
                    for kw in range(0, 3):
                        iw_k = ow_out + kw - 1
                        # compute input index
                        x_off = ((b * C_in + ic) * H + ih_k) * W + iw_k
                        # load x
                        x_val = tl.load(x_ptr + x_off).to(tl.float32)
                        # load weight w[co, ic, kh, kw]
                        w_off = co * (C_in * 9) + ic * 9 + kh * 3 + kw
                        w_val = tl.load(w_ptr + w_off).to(tl.float32)
                        acc += x_val * w_val
                # add bias
                b_val = tl.load(bias_ptr + co).to(tl.float32)
                y[ih_out * W_out + ow_out] = acc + b_val

    # Apply GELU (tanh approximation)
    c = 0.7978845608028654  # sqrt(2/pi)
    for pos in range(0, H_out * W_out):
        x = y[pos]
        x3 = x * x * x
        tanh_arg = c * (x + 0.044715 * x3)
        tanh_val = tl.math.tanh(tanh_arg)
        y[pos] = 0.5 * x * (1.0 + tanh_val)

    # Store to out[b, co, :, :]
    base_out = (b * C_out + co) * (H_out * W_out)
    for pos in range(0, H_out * W_out):
        tl.store(out_ptr + base_out + pos, y[pos])


@triton.jit
def linear_select3840_kernel(
    x_ptr,        # *f16/bf16, input: [B, T, 3840], contiguous
    w_ptr,        # *f16/bf16, weight: [1024, 3840], contiguous
    y_ptr,        # *f16/bf16, output: [B, T, 1024], contiguous
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr = 3840,  # we restrict to 3840 features (as in helper)
    N: tl.constexpr = 1024,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Reduce over K=3840
    for k0 in range(0, K):
        x_val = tl.load(x_ptr + b * (T * K) + t * K + k0).to(tl.float32)
        w_val = tl.load(w_ptr + d * K + k0).to(tl.float32)
        acc += x_val * w_val

    # Scale by embed_scale (32.0)
    acc = acc * 32.0
    tl.store(y_ptr + b * (T * N) + t * N + d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args from get_inputs: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale (ignored here)
        input_features = args[0]  # [B, 1, 80, time_dim], bfloat16
        conv2d1_weight = args[1]  # [384, 1, 3, 3], bfloat16
        conv2d1_bias = args[2]    # [384], bfloat16

        conv2d2_weight = args[3]  # [384, 1, 3, 3], bfloat16
        conv2d2_bias = args[4]    # [384], bfloat16

        conv2d3_weight = args[5]  # [384, 1, 3, 3], bfloat16
        conv2d3_bias = args[6]    # [384], bfloat16

        # Linear projection weight: helper sets conv_out_weight as [1024, 3840]
        conv_out_weight = args[7]  # shape [1024, 3840], bfloat16
        positional_embedding = args[8]  # [max_source_positions, 1024], bfloat16
        embed_scale = args[9]  # float

        device = input_features.device
        B, C_in, H, W = input_features.shape
        assert C_in == 1, "This Triton conv kernel expects C_in=1 as per the provided helper."

        # conv1
        C_out1 = 384
        H_out1 = (H + 2 * 1 - 3) // 2 + 1
        W_out1 = (W + 2 * 1 - 3) // 2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.bfloat16, device=device)
        grid1 = (B, C_out1)
        conv3x3_stride2_nchw_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
        )

        # conv2
        C_out2 = 384
        H2 = H_out1
        W2 = W_out1
        H_out2 = (H2 + 2 * 1 - 3) // 2 + 1
        W_out2 = (W2 + 2 * 1 - 3) // 2 + 1
        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.bfloat16, device=device)
        grid2 = (B, C_out2)
        conv3x3_stride2_nchw_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in, H2, W2, C_out2, H_out2, W_out2,
        )

        # conv3
        C_out3 = 384
        H3 = H_out2
        W3 = W_out2
        H_out3 = (H3 + 2 * 1 - 3) // 2 + 1
        W_out3 = (W3 + 2 * 1 - 3) // 2 + 1
        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.bfloat16, device=device)
        grid3 = (B, C_out3)
        conv3x3_stride2_nchw_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in, H3, W3, C_out3, H_out3, W_out3,
        )

        # Reshape: [B, W_out3, C_out3*H_out3] then view as [B, T=W_out3, K=C_out3*H_out3]
        T = W_out3
        K_full = C_out3 * H_out3 * W_out3

        # Flatten to [B, T, K_full]
        x3_contig = x3.contiguous()
        x_flat = x3_contig.view(B, T, K_full)

        # We need to compute only the first 3840 features (as in helper). Since K_full may be larger,
        # we select the first 3840 entries from x_flat along the last dimension and use conv_out_weight[:, :3840].
        x_select = x_flat[:, :, :3840]  # [B, T, 3840]
        conv_out_weight_select = conv_out_weight[:, :3840]  # [1024, 3840]

        # Launch linear kernel: outputs [B, T, 1024]
        y = torch.empty((B, T, 1024), dtype=torch.bfloat16, device=device)
        grid_lin = (B, T, 1024)
        linear_select3840_kernel[grid_lin](
            x_select, conv_out_weight_select, y,
            B, T,
        )

        # Scale by embed_scale
        y = y * embed_scale  # 32.0

        # Add positional embedding [:T, :] broadcast along batch
        pos_embed = positional_embedding[:T, :]  # [T, 1024], bfloat16
        y = y + pos_embed.unsqueeze(0)  # broadcast over batch

        return y


def run(*args):
    return ModelNew()(*args)
