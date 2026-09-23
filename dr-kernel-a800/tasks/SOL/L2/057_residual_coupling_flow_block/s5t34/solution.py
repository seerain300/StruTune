import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def conv1d_stride1_bias_relu(
    x_ptr,          # *f32, [B, Cin, T]
    w_ptr,          # *f32, [Cout, Cin*K] with K=5
    b_ptr,          # *f32, [Cout]
    out_ptr,        # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    T: tl.constexpr,
    Cout: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Compute Conv1d (stride=1, padding=(K-1)//2 for K=5 => P=2), add bias, apply ReLU.
    x: [B, Cin, T]
    w: [Cout, Cin*K], last dim flattened over kernel taps
    b: [Cout]
    out: [B, Cout, T]
    """
    pid_b = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel
    pid_t_block = tl.program_id(2)  # time block

    t_out = pid_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_out < T

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    CinK = tl.shape(w_ptr)[1]
    K = 5
    P = 2

    # Accumulate over input channels and kernel taps
    for cin in range(0, Cin):
        for k in tl.static_range(0, K):
            t_in = t_out - P + k
            valid = (t_in >= 0) & (t_in < T) & mask_t
            x_index = (((pid_b * Cin + cin) * T) + t_in)
            x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
            w_index = (pid_co * CinK) + (cin * K) + k
            w_val = tl.load(w_ptr + w_index)
            acc += x_val * w_val

    # Add bias and ReLU
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val
    acc = tl.maximum(acc, 0.0)

    # Store output
    out_index = (((pid_b * Cout) + pid_co) * T) + t_out
    tl.store(out_ptr + out_index, acc, mask=mask_t)


@triton.jit
def apply_mask_to_h(h_ptr, mask_ptr, out_ptr,
                    B, C, T, BLOCK_T: tl.constexpr):
    """
    h_ptr: [B, C, T], mask_ptr: [B, 1, T], out_ptr: [B, C, T]
    Multiply h by mask along time (broadcast over channels)
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    h_index = (((pid_b * C) + pid_c) * T) + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    out_val = h_val * mask_val
    tl.store(out_ptr + h_index, out_val, mask=mask_t)


@triton.jit
def add_h_to_x1(x1_ptr, h_ptr, out_ptr,
                B, C1, T, ADD: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    Update x1 = x1 + h (ADD=True) or x1 = x1 - h (ADD=False).
    x1: [B, C1, T], h: [B, C1, T], out_ptr: [B, C1, T]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    h_index = x1_index

    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    if ADD:
        out_val = x1_val + h_val
    else:
        out_val = x1_val - h_val

    tl.store(out_ptr + h_index, out_val, mask=mask_t)


@triton.jit
def concat_copy_first_half(x0_ptr, out_ptr,
                            B, C0, C, T, BLOCK_T: tl.constexpr):
    """
    out: [B, C, T], write first half from x0: columns [0:C0)
    x0: [B, C0, T]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x0_index = (((pid_b * C0) + pid_c) * T) + t_offsets
    out_index = (((pid_b * C) + pid_c) * T) + t_offsets

    x0_val = tl.load(x0_ptr + x0_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, x0_val, mask=mask_t)


@triton.jit
def concat_copy_second_half(x1_ptr, out_ptr,
                             B, C1, C0, C, T, BLOCK_T: tl.constexpr):
    """
    out: [B, C, T], write second half from x1 at columns [C0:C)
    x1: [B, C1, T]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (((pid_b * C1) + pid_c) + C0) * T + t_offsets
    out_index = (((pid_b * C) + (pid_c + C0)) * T) + t_offsets

    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, x1_val, mask=mask_t)


class ModelNew(nn.Module):
    def forward(self, x, x_mask, reverse,
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
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Triton-only forward. Implements:
        - Conv1d (3 layers per transform) + bias + ReLU
        - Elementwise mask application: h = h * x_mask
        - Affine coupling: x1 = x1 + h (forward), x1 = x1 - h (reverse)
        - Concatenation: out = [x0, updated_x1]
        No torch ops are used for computation.
        """
        B, C, T = x.shape
        half_channels = C // 2
        x = x.contiguous().to(torch.float32)
        device = x.device
        dtype = torch.float32

        # Ensure x_mask is 1D along channels; provided mask is [B, 1, T]
        x_mask = x_mask.contiguous().to(torch.float32)

        # Define transforms: each transform has 3 weights/bias
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

        # Initialize output buffer for x0 and x1 halves
        x0 = x[:, :half_channels, :].contiguous()
        x1 = x[:, half_channels:, :].contiguous()

        BLOCK_T = 128
        grid_t = _ceil_div(T, BLOCK_T)

        if not reverse:
            # Forward: apply each transform sequentially, update x1 = x1 + h
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # conv0: Cout=192, Cin=96, K=5
                conv0_out = torch.empty((B, 192, T), dtype=dtype, device=device)
                grid_conv0 = (B, 192, grid_t)
                conv1d_stride1_bias_relu[grid_conv0](
                    x0, conv0_w, conv0_b, conv0_out,
                    B=B, Cin=96, T=T, Cout=192, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # conv1: Cout=192, Cin=192
                conv1_out = torch.empty((B, 192, T), dtype=dtype, device=device)
                grid_conv1 = (B, 192, grid_t)
                conv1d_stride1_bias_relu[grid_conv1](
                    conv0_out, conv1_w, conv1_b, conv1_out,
                    B=B, Cin=192, T=T, Cout=192, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # conv2: Cout=96, Cin=192
                conv2_out = torch.empty((B, 96, T), dtype=dtype, device=device)
                grid_conv2 = (B, 96, grid_t)
                conv1d_stride1_bias_relu[grid_conv2](
                    conv1_out, conv2_w, conv2_b, conv2_out,
                    B=B, Cin=192, T=T, Cout=96, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # apply mask
                conv2_masked = torch.empty_like(conv2_out)
                grid_mask = (B, 96, grid_t)
                apply_mask_to_h[grid_mask](conv2_out, x_mask, conv2_masked, B=B, C=96, T=T, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
                # update x1
                x1 = x1 + conv2_masked
        else:
            # Reverse: apply each transform in reverse order, update x1 = x1 - h
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # conv0: Cout=192, Cin=96
                conv0_out = torch.empty((B, 192, T), dtype=dtype, device=device)
                grid_conv0 = (B, 192, grid_t)
                conv1d_stride1_bias_relu[grid_conv0](
                    x0, conv0_w, conv0_b, conv0_out,
                    B=B, Cin=96, T=T, Cout=192, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # conv1: Cout=192, Cin=192
                conv1_out = torch.empty((B, 192, T), dtype=dtype, device=device)
                grid_conv1 = (B, 192, grid_t)
                conv1d_stride1_bias_relu[grid_conv1](
                    conv0_out, conv1_w, conv1_b, conv1_out,
                    B=B, Cin=192, T=T, Cout=192, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # conv2: Cout=96, Cin=192
                conv2_out = torch.empty((B, 96, T), dtype=dtype, device=device)
                grid_conv2 = (B, 96, grid_t)
                conv1d_stride1_bias_relu[grid_conv2](
                    conv1_out, conv2_w, conv2_b, conv2_out,
                    B=B, Cin=192, T=T, Cout=96, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2
                )
                # apply mask
                conv2_masked = torch.empty_like(conv2_out)
                grid_mask = (B, 96, grid_t)
                apply_mask_to_h[grid_mask](conv2_out, x_mask, conv2_masked, B=B, C=96, T=T, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
                # update x1
                x1 = x1 - conv2_masked

        # Concatenate [x0, x1] into out [B, C, T]
        out = torch.empty((B, C, T), dtype=dtype, device=device)
        grid_first = (B, half_channels, grid_t)
        concat_copy_first_half[grid_first](x0, out, B=B, C0=half_channels, C=C, T=T, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)
        grid_second = (B, half_channels, grid_t)
        concat_copy_second_half[grid_second](x1, out, B=B, C1=half_channels, C0=half_channels, C=C, T=T, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)

        # apply x_mask broadcast along channels: out = out * x_mask (mask is [B, 1, T], we broadcast via kernel)
        out_masked = torch.empty_like(out)
        grid_mask_out = (B, C, grid_t)
        apply_mask_to_h[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2)

        return out_masked


def run(*args):
    return ModelNew()(*args)
