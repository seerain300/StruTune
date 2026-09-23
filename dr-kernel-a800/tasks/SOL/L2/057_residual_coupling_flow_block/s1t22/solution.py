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
    Triton Conv1d (cross-correlation) for stride=1, padding=pad, dilation=1.
    x: [N, IC, T], w: [OC, IC, K], out: [N, OC, T].
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
def mask_apply_to_h_triton(h_ptr, mask_ptr, out_ptr,
                           N, OC, T,
                           h_sN, h_sC, h_sT,
                           mask_sN, mask_sC, mask_sT,  # mask shape [N, 1, T]
                           out_sN, out_sC, out_sT,
                           BLOCK: tl.constexpr):
    """
    Multiply h [N, OC, T] elementwise by mask [N, 1, T], broadcast along OC, store to out_ptr.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_h = pid_n * h_sN + pid_oc * h_sC
    base_out = pid_n * out_sN + pid_oc * out_sC
    mask_off = pid_n * mask_sN  # C dimension is 1, so ignore mask_sC

    h_vals = tl.load(h_ptr + base_h + idx * h_sT, mask=valid, other=0.0)
    m_vals = tl.load(mask_ptr + mask_off + idx * mask_sT, mask=valid, other=1.0)
    h_vals = h_vals * m_vals
    tl.store(out_ptr + base_out + idx * out_sT, h_vals, mask=valid)


@triton.jit
def add_h_to_second_half_triton(x_ptr, h_ptr, out_ptr,
                                 N, C, T,
                                 x_sN, x_sC, x_sT,
                                 h_sN, h_sC, h_sT,
                                 out_sN, out_sC, out_sT,
                                 half_channels, reverse: tl.int32,
                                 BLOCK: tl.constexpr):
    """
    Update second half channels: out = x1 + h if reverse==0 else out = x1 - h.
    x_ptr points to the entire x, we read/write channels [half_channels:C).
    h has shape [N, half_channels, T] (conv2 output).
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [half_channels, C)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_x = pid_n * x_sN + pid_c * x_sC
    base_out = pid_n * out_sN + pid_c * out_sC
    base_h = pid_n * h_sN + (pid_c - half_channels) * h_sC  # h has OC=half_channels

    x_vals = tl.load(x_ptr + base_x + idx * x_sT, mask=valid, other=0.0)
    h_vals = tl.load(h_ptr + base_h + idx * h_sT, mask=valid, other=0.0)
    if reverse == 0:
        new_vals = x_vals + h_vals
    else:
        new_vals = x_vals - h_vals
    tl.store(out_ptr + base_out + idx * out_sT, new_vals, mask=valid)


@triton.jit
def copy_second_half_triton(x_full_ptr, x_half_ptr,
                             N, C, T,
                             x_full_sN, x_full_sC, x_full_sT,
                             x_half_sN, x_half_sC, x_half_sT,
                             half_channels, BLOCK: tl.constexpr):
    """
    Copy x_half_ptr [N, half_channels, T] into x_full_ptr at channels [half_channels:C).
    Grid: (N, C - half_channels, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [half_channels, C)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_src = pid_n * x_half_sN + (pid_c - half_channels) * x_half_sC
    base_dst = pid_n * x_full_sN + pid_c * x_full_sC

    vals = tl.load(x_half_ptr + base_src + idx * x_half_sT, mask=valid, other=0.0)
    tl.store(x_full_ptr + base_dst + idx * x_full_sT, vals, mask=valid)


@triton.jit
def copy_first_half_triton(x_full_ptr, x_half_ptr,
                            N, C, T,
                            x_full_sN, x_full_sC, x_full_sT,
                            x_half_sN, x_half_sC, x_half_sT,
                            half_channels, BLOCK: tl.constexpr):
    """
    Copy x_half_ptr [N, half_channels, T] into x_full_ptr at channels [0:half_channels).
    Grid: (N, half_channels, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [0, half_channels)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_src = pid_n * x_half_sN + pid_c * x_half_sC
    base_dst = pid_n * x_full_sN + pid_c * x_full_sC

    vals = tl.load(x_half_ptr + base_src + idx * x_half_sT, mask=valid, other=0.0)
    tl.store(x_full_ptr + base_dst + idx * x_full_sT, vals, mask=valid)


@triton.jit
def multiply_mask_over_all_channels_triton(x_ptr, mask_ptr, out_ptr,
                                           N, C, T,
                                           x_sN, x_sC, x_sT,
                                           mask_sN, mask_sC, mask_sT,  # mask shape [N, 1, T]
                                           out_sN, out_sC, out_sT,
                                           BLOCK: tl.constexpr):
    """
    Multiply x [N, C, T] elementwise by mask [N, 1, T], broadcast along C, store to out_ptr.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)
    idx = pid_block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < T

    base_x = pid_n * x_sN + pid_c * x_sC
    base_out = pid_n * out_sN + pid_c * out_sC
    mask_off = pid_n * mask_sN  # C dimension is 1

    x_vals = tl.load(x_ptr + base_x + idx * x_sT, mask=valid, other=0.0)
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
    Triton-only implementation: forward with coupling and mask scaling.
    - No torch.conv1d or torch.cat in forward.
    - Launch Triton kernels for conv, ReLU, mask apply, coupling, and final scaling.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 (half_channels=96)."
    half_channels = C // 2

    # Process transforms
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
        # Split into halves
        x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
        x1 = x[:, half_channels:, :].contiguous() # [N, 96, T]

        # conv0: out_channels=192, in_channels=96, K=5, padding=2
        OC0 = 192
        IC0 = 96
        K0 = 5
        pad0 = K0 // 2
        y0 = torch.empty((N, OC0, T), dtype=x.dtype, device=x.device)
        grid_conv0 = (N, OC0, triton.cdiv(T, 128))
        conv1d_triton[grid_conv0](
            x0, w0, b0, y0,
            N, IC0, OC0, T, K0, pad0,
            x0.stride(0), x0.stride(1), x0.stride(2),
            w0.stride(0), w0.stride(1), w0.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # ReLU on y0
        y0_relu = torch.empty_like(y0)
        grid_relu0 = (N, OC0, triton.cdiv(T, 128))
        relu_triton[grid_relu0](
            y0, y0_relu, N, OC0, T,
            y0.stride(0), y0.stride(1), y0.stride(2),
            y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv1: out_channels=192, in_channels=192, K=5, padding=2
        OC1 = 192
        IC1 = 192
        K1 = 5
        pad1 = K1 // 2
        y1 = torch.empty((N, OC1, T), dtype=x.dtype, device=x.device)
        grid_conv1 = (N, OC1, triton.cdiv(T, 128))
        conv1d_triton[grid_conv1](
            y0_relu, w1, b1, y1,
            N, IC1, OC1, T, K1, pad1,
            y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            w1.stride(0), w1.stride(1), w1.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # ReLU on y1
        y1_relu = torch.empty_like(y1)
        grid_relu1 = (N, OC1, triton.cdiv(T, 128))
        relu_triton[grid_relu1](
            y1, y1_relu, N, OC1, T,
            y1.stride(0), y1.stride(1), y1.stride(2),
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv2: out_channels=half_channels=96, in_channels=192, K=5, padding=2
        OC2 = half_channels
        IC2 = 192
        K2 = 5
        pad2 = K2 // 2
        h = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)
        grid_conv2 = (N, OC2, triton.cdiv(T, 128))
        conv1d_triton[grid_conv2](
            y1_relu, w2, (b2 if b2 is not None else torch.zeros(OC2, dtype=x.dtype, device=x.device)), h,
            N, IC2, OC2, T, K2, pad2,
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # Apply mask to h (broadcast along channels)
        h_masked = torch.empty_like(h)
        grid_mask_h = (N, OC2, triton.cdiv(T, 128))
        mask_apply_to_h_triton[grid_mask_h](
            h, x_mask, h_masked,
            N, OC2, T,
            h.stride(0), h.stride(1), h.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # Update second half channels: x1 = x1 + h or x1 = x1 - h
        # Launch add_h_to_second_half_triton; grid over (N, half_channels, ceil(T/128))
        grid_add = (N, half_channels, triton.cdiv(T, 128))
        add_h_to_second_half_triton[grid_add](
            x, h_masked, x,  # write back into x
            N, C, T,
            x.stride(0), x.stride(1), x.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            half_channels, 0 if not reverse else 1,
            BLOCK=128, num_warps=4
        )

        # Finally, scale the entire x by x_mask (broadcast along channels)
        x_mask_expanded = x_mask  # shape [N, 1, T]
        x_scaled = torch.empty_like(x)
        grid_scale = (N, C, triton.cdiv(T, 128))
        multiply_mask_over_all_channels_triton[grid_scale](
            x, x_mask_expanded, x_scaled,
            N, C, T,
            x.stride(0), x.stride(1), x.stride(2),
            x_mask_expanded.stride(0), x_mask_expanded.stride(1), x_mask_expanded.stride(2),
            x_scaled.stride(0), x_scaled.stride(1), x_scaled.stride(2),
            BLOCK=128, num_warps=4
        )
        # Update x for next iteration
        x = x_scaled

    return x

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The first arg is x, the second is x_mask, the third is reverse flag.
        # Remaining args are weights/biases.
        # We rely on ModelNew.forward to launch Triton kernels and avoid torch ops.
        x, x_mask, reverse = args[0], args[1], args[2]
        weights = args[3:]
        return run(x, x_mask, reverse, *weights)


def run(*args):
    return ModelNew()(*args)
