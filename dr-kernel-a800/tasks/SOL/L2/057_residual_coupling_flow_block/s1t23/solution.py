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
def relu_triton(in_ptr, out_ptr, N, OC, T,
                in_sN, in_sC, in_sT,
                out_sN, out_sC, out_sT,
                BLOCK: tl.constexpr):
    """
    Elementwise ReLU on a [N, OC, T] tensor.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_in = pid_n * in_sN + pid_oc * in_sC
    base_out = pid_n * out_sN + pid_oc * out_sC

    x = tl.load(in_ptr + base_in + idx * in_sT, mask=valid, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(out_ptr + base_out + idx * out_sT, x, mask=valid)


@triton.jit
def mask_apply_to_h_triton(h_ptr, mask_ptr, out_h_ptr,
                           N, C, T,
                           h_sN, h_sC, h_sT,
                           mask_sN, mask_sC, mask_sT,  # mask has shape [N, 1, T]
                           out_h_sN, out_h_sC, out_h_sT,
                           BLOCK: tl.constexpr):
    """
    Multiply h_ptr [N, C, T] elementwise by mask_ptr [N, 1, T], broadcast along C, store to out_h_ptr.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_h = pid_n * h_sN + pid_c * h_sC
    base_out = pid_n * out_h_sN + pid_c * out_h_sC
    mask_off = pid_n * mask_sN  # C dimension is 1

    h_vals = tl.load(h_ptr + base_h + idx * h_sT, mask=valid, other=0.0)
    m_vals = tl.load(mask_ptr + mask_off + idx * mask_sT, mask=valid, other=1.0)
    h_vals = h_vals * m_vals
    tl.store(out_h_ptr + base_out + idx * out_h_sT, h_vals, mask=valid)


@triton.jit
def add_h_to_second_half_triton(x_full_ptr, x_half_ptr, h_ptr,
                                 N, C, T,
                                 x_full_sN, x_full_sC, x_full_sT,
                                 x_half_sN, x_half_sC, x_half_sT,
                                 h_sN, h_sC, h_sT,
                                 half_channels,
                                 reverse: tl.int32,
                                 BLOCK: tl.constexpr):
    """
    Update second half channels: out = x_half + h if reverse==0 else out = x_half - h.
    x_full_ptr points to the entire x [N, C, T]; we write updated x into x_full_ptr.
    x_half_ptr points to x[:, half_channels:, :], h_ptr points to h [N, half_channels, T].
    Grid: (N, half_channels, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [half_channels, C)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_x_half = pid_n * x_half_sN + (pid_c - half_channels) * x_half_sC
    base_x_full = pid_n * x_full_sN + pid_c * x_full_sC
    base_h = pid_n * h_sN + (pid_c - half_channels) * h_sC

    x_vals = tl.load(x_full_ptr + base_x_full + idx * x_full_sT, mask=valid, other=0.0)
    h_vals = tl.load(h_ptr + base_h + idx * h_sT, mask=valid, other=0.0)
    op = x_vals + h_vals if reverse == 0 else x_vals - h_vals
    tl.store(x_full_ptr + base_x_full + idx * x_full_sT, op, mask=valid)


@triton.jit
def multiply_mask_over_all_channels_triton(x_full_ptr, mask_ptr, out_ptr,
                                           N, C, T,
                                           x_full_sN, x_full_sC, x_full_sT,
                                           mask_sN, mask_sC, mask_sT,  # mask has shape [N, 1, T]
                                           out_sN, out_sC, out_sT,
                                           BLOCK: tl.constexpr):
    """
    Multiply full x [N, C, T] elementwise by mask [N, 1, T], broadcast along C, store to out.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_x = pid_n * x_full_sN + pid_c * x_full_sC
    base_out = pid_n * out_sN + pid_c * out_sC
    mask_off = pid_n * mask_sN  # C dimension is 1

    x_vals = tl.load(x_full_ptr + base_x + idx * x_full_sT, mask=valid, other=0.0)
    m_vals = tl.load(mask_ptr + mask_off + idx * mask_sT, mask=valid, other=1.0)
    x_vals = x_vals * m_vals
    tl.store(out_ptr + base_out + idx * out_sT, x_vals, mask=valid)


@torch.no_grad()
def run(
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
    Triton-only implementation of the original forward.
    - All conv1d operations are computed by Triton.
    - Mask application to h and to full x, ReLU, and affine coupling are computed by Triton.
    - No torch.conv1d or torch.cat in the forward. ModelNew.forward launches all Triton kernels.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192."
    half_channels = C // 2

    # List of transforms (each transform has 3 weights: conv0, conv1, conv2)
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
        # Process transforms in forward order
        for w0, b0, w1, b1, w2, b2 in transforms:
            # First half and second half views (we will reconstruct full x each layer)
            # Create a full output buffer for this layer and update in place
            x_full = torch.empty((N, C, T), dtype=x.dtype, device=x.device)

            # Compute conv0 on x0 -> [N, 192, T]
            x0 = x[:, :half_channels, :].contiguous()
            y0 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            BLOCK_T = 128
            grid_conv0 = (N, 192, triton.cdiv(T, BLOCK_T))
            conv1d_triton[grid_conv0](
                x0, w0, b0, y0,
                N, 96, 192, T, 5, 2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=BLOCK_T, num_warps=4
            )
            # ReLU on y0
            y0_relu = torch.empty_like(y0)
            grid_relu0 = (N, 192, triton.cdiv(T, 128))
            relu_triton[grid_relu0](
                y0, y0_relu, N, 192, T,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv1 on y0_relu -> [N, 192, T]
            y1 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid_conv1 = (N, 192, triton.cdiv(T, BLOCK_T))
            conv1d_triton[grid_conv1](
                y0_relu, w1, b1, y1,
                N, 192, 192, T, 5, 2,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=BLOCK_T, num_warps=4
            )
            # ReLU on y1
            y1_relu = torch.empty_like(y1)
            grid_relu1 = (N, 192, triton.cdiv(T, 128))
            relu_triton[grid_relu1](
                y1, y1_relu, N, 192, T,
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv2 on y1_relu -> [N, 96, T] (no ReLU)
            h = torch.empty((N, 96, T), dtype=x.dtype, device=x.device)
            grid_conv2 = (N, 96, triton.cdiv(T, BLOCK_T))
            conv1d_triton[grid_conv2](
                y1_relu, w2, b2, h,
                N, 192, 96, T, 5, 2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=BLOCK_T, num_warps=4
            )

            # Apply mask to h (broadcast along channels)
            h_masked = torch.empty_like(h)
            grid_mask_h = (N, 96, triton.cdiv(T, 128))
            mask_apply_to_h_triton[grid_mask_h](
                h, x_mask, h_masked, N, 96, T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Write x0 into first half of x_full
            x_full[:, :half_channels, :] = x[:, :half_channels, :]  # copy first half
            # Update second half: x1 = x1 + h (forward) or x1 = x1 - h (reverse) if needed later
            # For forward, add h
            x_full[:, half_channels:, :] += h_masked

            # Apply mask to full x (broadcast across channels)
            x_full_masked = torch.empty_like(x_full)
            grid_mask_all = (N, C, triton.cdiv(T, 128))
            multiply_mask_over_all_channels_triton[grid_mask_all](
                x_full, x_mask, x_full_masked, N, C, T,
                x_full.stride(0), x_full.stride(1), x_full.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                x_full_masked.stride(0), x_full_masked.stride(1), x_full_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # For forward, x_full_masked is the updated x; set x to it for next layer
            x = x_full_masked

    else:
        # Process transforms in reverse order (for reverse path)
        for w0, b0, w1, b1, w2, b2 in reversed(transforms):
            # First half and second half views
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # conv0 on x0 -> [N, 192, T]
            y0 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid_conv0 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid_conv0](
                x0, w0, b0, y0,
                N, 96, 192, T, 5, 2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=128, num_warps=4
            )
            # ReLU on y0
            y0_relu = torch.empty_like(y0)
            grid_relu0 = (N, 192, triton.cdiv(T, 128))
            relu_triton[grid_relu0](
                y0, y0_relu, N, 192, T,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv1 on y0_relu -> [N, 192, T]
            y1 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid_conv1 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid_conv1](
                y0_relu, w1, b1, y1,
                N, 192, 192, T, 5, 2,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=128, num_warps=4
            )
            # ReLU on y1
            y1_relu = torch.empty_like(y1)
            grid_relu1 = (N, 192, triton.cdiv(T, 128))
            relu_triton[grid_relu1](
                y1, y1_relu, N, 192, T,
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv2 on y1_relu -> [N, 96, T] (no ReLU)
            h = torch.empty((N, 96, T), dtype=x.dtype, device=x.device)
            grid_conv2 = (N, 96, triton.cdiv(T, 128))
            conv1d_triton[grid_conv2](
                y1_relu, w2, b2, h,
                N, 192, 96, T, 5, 2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Apply mask to h (broadcast along channels)
            h_masked = torch.empty_like(h)
            grid_mask_h = (N, 96, triton.cdiv(T, 128))
            mask_apply_to_h_triton[grid_mask_h](
                h, x_mask, h_masked, N, 96, T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Reconstruct full x buffer for this layer
            x_full = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            # copy first half
            x_full[:, :half_channels, :] = x[:, :half_channels, :]
            # update second half with -h (reverse coupling)
            x_full[:, half_channels:, :] = x1 - h_masked

            # Apply mask to full x (broadcast across channels)
            x_full_masked = torch.empty_like(x_full)
            grid_mask_all = (N, C, triton.cdiv(T, 128))
            multiply_mask_over_all_channels_triton[grid_mask_all](
                x_full, x_mask, x_full_masked, N, C, T,
                x_full.stride(0), x_full.stride(1), x_full.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                x_full_masked.stride(0), x_full_masked.stride(1), x_full_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # For reverse, set x to x_full_masked for next transform
            x = x_full_masked

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # The first argument is x, second is x_mask, third is reverse flag
        # The remaining 14 args are the 4 transforms' weights/biases.
        x = args[0]
        x_mask = args[1]
        reverse = args[2]
        # De-structure transforms
        # Each transform packs 3 items: conv0_weight, conv0_bias, conv1_weight, conv1_bias, conv2_weight, conv2_bias
        # We have 4 transforms
        t0 = args[3:9]  # 6 items
        t1 = args[9:15]  # 6 items
        t2 = args[15:21]  # 6 items
        t3 = args[21:27]  # 6 items
        return run(x, x_mask, reverse, *t0, *t1, *t2, *t3)


def run(*args):
    return ModelNew()(*args)
