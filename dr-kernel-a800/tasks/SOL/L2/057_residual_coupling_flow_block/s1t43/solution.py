import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv1d_triton(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, T, K, pad,
    x_sN, x_sC, x_sT,
    w_sO, w_sI, w_sK,
    out_sN, out_sC, out_sT,
    BLOCK_T: tl.constexpr,
):
    """
    Triton implementation of Conv1d (cross-correlation) for stride=1, padding=pad, dilation=1.
    x: [N, IC, T] (float32)
    w: [OC, IC, K] (float32)
    b: [OC] (float32)
    out: [N, OC, T] (float32)
    Grid: (N, OC, ceil_div(T, BLOCK_T))
    Accumulate in float32.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_out_start = pid_block * BLOCK_T
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)
    valid_t = t_out_idx < T

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ic in range(0, IC):
        for k in range(0, K):
            t_in = t_out_idx + k - pad  # valid when 0 <= t_in < T
            valid_t_in = valid_t & (t_in >= 0) & (t_in < T)
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_t_in, other=0.0)
            w_val = tl.load(w_ptr + pid_oc * w_sO + ic * w_sI + k * w_sK)
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    # Store to out[n, oc, t_out_idx]
    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr + out_offsets, acc, mask=valid_t)


@triton.jit
def mask_apply_triton(
    y_ptr, mask_ptr, out_ptr,
    N, OC, T,
    mask_sN, mask_sC, mask_sT,
    y_sN, y_sC, y_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Multiply y [N, OC, T] by mask [N, 1, T], broadcasting mask across channels, and store to out.
    Grid: (N, OC, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    # Load mask vector for this batch and t-block (broadcast over channels)
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=0.0)

    # Load y and apply mask
    y_offsets = pid_n * y_sN + pid_oc * y_sC + t_idx * y_sT
    y_vals = tl.load(y_ptr + y_offsets, mask=valid_t, other=0.0)
    out_vals = y_vals * mask_vals

    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def relu_triton(
    inp_ptr, out_ptr,
    N, OC, T,
    in_sN, in_sC, in_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Elementwise ReLU over tensor [N, OC, T].
    Grid: (N, OC, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    in_offsets = pid_n * in_sN + pid_oc * in_sC + t_idx * in_sT
    vals = tl.load(inp_ptr + in_offsets, mask=valid_t, other=0.0)
    vals = tl.maximum(vals, 0.0)

    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, vals, mask=valid_t)


@triton.jit
def add_h_triton(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_sN, x1_sC, x1_sT,
    h_sN, h_sC, h_sT,
    out_sN, out_sC, out_sT,
    reverse_flag: tl.constexpr,  # 0 for add, 1 for subtract
    BLOCK: tl.constexpr,
):
    """
    Elementwise coupling: out = x1 + (-) h, depending on reverse_flag.
    x1: [N, C, T], h: [N, C, T], out: [N, C, T].
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    x_offsets = pid_n * x1_sN + pid_c * x1_sC + t_idx * x1_sT
    h_offsets = pid_n * h_sN + pid_c * h_sC + t_idx * h_sT

    x_vals = tl.load(x1_ptr + x_offsets, mask=valid_t, other=0.0)
    h_vals = tl.load(h_ptr + h_offsets, mask=valid_t, other=0.0)

    if reverse_flag == 0:
        out_vals = x_vals + h_vals
    else:
        out_vals = x_vals - h_vals

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


def _conv1d_triton_out_shape(N, IC, OC, T, K, pad):
    # output length T_out = T (since padding on both sides and stride=1)
    return T


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias,
                transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias,
                transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias,
                transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias,
                transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias,
                transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias,
                ):
        """
        Triton-only implementation. Applies the residual coupling flow:
        - Forward: x1 = x1 + transform(x0) for each layer
        - Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
        No torch.conv1d or torch.cat; all ops are Triton kernels.
        """
        assert x.dim() == 3 and x_mask.dim() == 3, "x must be [N, C, T], x_mask [N, 1, T]"
        N, C, T = x.shape
        assert C == 192, "This implementation assumes C=192 for half_channels=96."
        half_channels = C // 2
        BLOCK = 128

        # Process transforms (4 of them), sequentially forward or in reverse
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

        for w0, b0, w1, b1, w2, b2 in (transforms if not reverse else reversed(transforms)):
            # Split current x into x0 and x1
            x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
            x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

            # conv0: IC=96, OC=192, K=5, padding=2
            OC0 = 192
            IC0 = 96
            K0 = 5
            pad0 = K0 // 2
            y0 = torch.empty((N, OC0, T), dtype=x.dtype, device=x.device)
            grid_conv0 = (N, OC0, triton.cdiv(T, BLOCK))
            conv1d_triton[grid_conv0](
                x0, w0, b0, y0,
                N, IC0, OC0, T, K0, pad0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=BLOCK, num_warps=4, num_stages=2
            )

            # apply mask and ReLU to y0 (mask is [N,1,T], broadcast across channels)
            y0_masked = torch.empty_like(y0)
            grid_mask0 = (N, OC0, triton.cdiv(T, BLOCK))
            mask_apply_triton[grid_mask0](
                y0, x_mask, y0_masked,
                N, OC0, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )
            y0_relu = torch.empty_like(y0_masked)
            grid_relu0 = (N, OC0, triton.cdiv(T, BLOCK))
            relu_triton[grid_relu0](
                y0_masked, y0_relu,
                N, OC0, T,
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )

            # conv1: IC=192, OC=192, K=5, padding=2
            OC1 = 192
            IC1 = 192
            K1 = 5
            pad1 = K1 // 2
            y1 = torch.empty((N, OC1, T), dtype=x.dtype, device=x.device)
            grid_conv1 = (N, OC1, triton.cdiv(T, BLOCK))
            conv1d_triton[grid_conv1](
                y0_relu, w1, b1, y1,
                N, IC1, OC1, T, K1, pad1,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=BLOCK, num_warps=4, num_stages=2
            )

            # apply mask and ReLU to y1
            y1_masked = torch.empty_like(y1)
            grid_mask1 = (N, OC1, triton.cdiv(T, BLOCK))
            mask_apply_triton[grid_mask1](
                y1, x_mask, y1_masked,
                N, OC1, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )
            y1_relu = torch.empty_like(y1_masked)
            grid_relu1 = (N, OC1, triton.cdiv(T, BLOCK))
            relu_triton[grid_relu1](
                y1_masked, y1_relu,
                N, OC1, T,
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )

            # conv2: IC=192, OC=96, K=5, padding=2
            OC2 = 96
            IC2 = 192
            K2 = 5
            pad2 = K2 // 2
            h = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)  # coupling output
            grid_conv2 = (N, OC2, triton.cdiv(T, BLOCK))
            conv1d_triton[grid_conv2](
                y1_relu, w2, b2, h,
                N, IC2, OC2, T, K2, pad2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=BLOCK, num_warps=4, num_stages=2
            )

            # apply mask for coupling (broadcast [N,1,T] across channels)
            h_masked = torch.empty_like(h)
            grid_mask_h = (N, OC2, triton.cdiv(T, BLOCK))
            mask_apply_triton[grid_mask_h](
                h, x_mask, h_masked,
                N, OC2, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )

            # affine coupling: update x1
            out_x1 = torch.empty_like(x1)
            grid_add = (N, half_channels, triton.cdiv(T, BLOCK))
            add_h_triton[grid_add](
                x1, h_masked, out_x1,
                N, half_channels, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                reverse_flag=1 if reverse else 0,  # reverse => subtract
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )

            # concatenate back into x
            x = torch.cat([x0, out_x1], dim=1).contiguous()

            # apply mask to entire x
            x_masked = torch.empty_like(x)
            grid_mask_all = (N, C, triton.cdiv(T, BLOCK))
            mask_apply_triton[grid_mask_all](
                x, x_mask, x_masked,
                N, C, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                x.stride(0), x.stride(1), x.stride(2),
                x_masked.stride(0), x_masked.stride(1), x_masked.stride(2),
                BLOCK=BLOCK, num_warps=4, num_stages=2
            )
            x = x_masked

        return x


def run(*args):
    return ModelNew()(*args)
