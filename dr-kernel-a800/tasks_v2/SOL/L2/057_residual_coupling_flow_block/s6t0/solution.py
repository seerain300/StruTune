import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_k5_p2(
    x_ptr,        # *float32, [B, C_in, T_in]
    w_ptr,        # *float32, [C_out, C_in, 5]
    y_ptr,        # *float32, [B, C_out, T_out] where T_out = T_in - 1 (padding=2, kernel=5)
    B, C_out, C_in, T_in, T_out,
    stride_xb, stride_xc, stride_xt,
    stride_wco, stride_wci, stride_wk,
    stride_yb, stride_yco, stride_yt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C_out, ceil_div(T_out, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decode batch and output channel
    b = pid0 // C_out
    co = pid0 % C_out

    # Time tile
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T_out

    # Accumulator for output vector across BLOCK_T
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # For each input channel and kernel tap, compute contribution
    # Padding=2: output index t_out corresponds to input index t_in = t_out - 2 + k
    for ci in range(0, C_in):
        for k in range(0, 5):
            t_in = t_offsets - 2 + k  # valid since T_out = T_in - 1
            # Load x[b, ci, t_in]
            x_idx = b * stride_xb + ci * stride_xc + t_in * stride_xt
            x_vals = tl.load(x_ptr + x_idx, mask=mask_t, other=0.0)
            # Load w[co, ci, k]
            w_idx = co * stride_wco + ci * stride_wci + k * stride_wk
            w_val = tl.load(w_ptr + w_idx)
            acc += x_vals * w_val

    # Store result to y[b, co, t_offsets]
    y_idx = b * stride_yb + co * stride_yco + t_offsets * stride_yt
    tl.store(y_ptr + y_idx, acc, mask=mask_t)


@triton.jit
def conv1d_bias(
    y_ptr,        # *float32, [B, C_out, T_out] (pre-conv output)
    bias_ptr,     # *float32, [C_out]
    B, C_out, T_out,
    stride_yb, stride_yco, stride_yt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C_out, ceil_div(T_out, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // C_out
    co = pid0 % C_out

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T_out

    y_idx = b * stride_yb + co * stride_yco + t_offsets * stride_yt
    y_vals = tl.load(y_ptr + y_idx, mask=mask_t, other=0.0)

    bias_val = tl.load(bias_ptr + co)
    y_vals += bias_val

    tl.store(y_ptr + y_idx, y_vals, mask=mask_t)


class ModelNew(torch.nn.Module):
    def forward(self, x, x_mask, reverse,
                # transform 0
                transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias,
                transform_0_conv2_weight, transform_0_conv2_bias,
                # transform 1
                transform_1_conv0_weight, transform_1_conv0_bias,
                transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias,
                # transform 2
                transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias,
                transform_2_conv2_weight, transform_2_conv2_bias,
                # transform 3
                transform_3_conv0_weight, transform_3_conv0_bias,
                transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Triton-optimized forward and reverse pass. All convs are done by Triton kernels.
        Elementwise ops (mask multiply, add/sub, concat) are done in PyTorch.
        """
        device = x.device
        half_channels = x.shape[1] // 2

        # List of transforms (each is (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))
        transforms = [
            (transform_0_conv0_weight, transform_0_conv0_bias,
             transform_0_conv1_weight, transform_0_conv1_bias,
             transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias,
             transform_1_conv1_weight, transform_1_conv1_bias,
             transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias,
             transform_2_conv1_weight, transform_2_conv1_bias,
             transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias,
             transform_3_conv1_weight, transform_3_conv1_bias,
             transform_3_conv2_weight, transform_3_conv2_bias),
        ]

        if not reverse:
            # Forward: apply transforms sequentially
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Split input
                x0 = x[:, :half_channels, :]
                x1 = x[:, half_channels:, :]

                hidden_channels = conv0_w.shape[0]

                T_in0 = x0.shape[2]  # original time length
                # With padding=2 and kernel=5, output length is T_in - 1
                T_out0 = T_in0 - 1

                # conv0
                h = self._triton_conv1d_k5_p2(x0, conv0_w, T_out=T_out0)
                # bias and ReLU
                h = self._triton_add_bias(h, conv0_b)
                h = F.relu(h)  # elementwise ReLU in PyT


def run(*args):
    return ModelNew()(*args)
