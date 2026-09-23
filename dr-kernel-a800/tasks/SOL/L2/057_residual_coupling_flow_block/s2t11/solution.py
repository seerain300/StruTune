import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def concat_halves_backward(
    y2c_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    # y2c has shape [N, 2*C_half, L]; split along channel dimension
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y0_ptrs = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y1_ptrs = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l
    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(y0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(y1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y2c_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    BLOCK_L: tl.constexpr,
):
    # y0: [N, C_half, L], y1: [N, C_half, L]; write y2c: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    y2c_ptrs0 = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y2c_ptrs1 = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l

    y0_vals = tl.load(out0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(out1_ptrs, mask=mask_out, other=0.0)
    tl.store(y2c_ptrs0, y0_vals, mask=mask_out)
    tl.store(y2c_ptrs1, y1_vals, mask=mask_out)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,  # mask shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_m_n + 0 * stride_m_c + l_offsets * stride_m_l

    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_out, other=1.0)
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


def single_transform_triton(
    x_curr,                 # [N, C, L] current x (to split into halves)
    mask,                   # [N, 1, L] x_mask
    w0, b0,                 # conv0 weights/bias
    w1, b1,                 # conv1 weights/bias
    w2, b2,                 # conv2 weights/bias
    reverse: bool,          # whether to subtract h
    device,                 # Triton requires CUDA tensors
):
    # Ensure contiguous for predictable strides
    x_curr = x_curr.contiguous()
    N, C, L = x_curr.shape
    half = C // 2

    # Split into x0 and x1
    x0 = torch.empty((N, half, L), device=device, dtype=x_curr.dtype)
    x1 = torch.empty((N, half, L), device=device, dtype=x_curr.dtype)

    grid_split = (N, half, triton.cdiv(L, 128))
    concat_halves_backward[grid_split](
        x_curr, x0, x1,
        N, half, L,
        x_curr.stride(0), x_curr.stride(1), x_curr.stride(2),
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Compute h = apply_transform(x0) using PyTorch conv1d (padding = kernel_size // 2)
    k = w0.shape[2]
    padding = k // 2

    # conv0
    h0 = F.conv1d(x0, w0, b0, padding=padding)  # [N, hidden_channels, L]
    h0 = F.relu(h0)  # ReLU after conv0

    # conv1
    h1 = F.conv1d(h0, w1, b1, padding=padding)  # [N, hidden_channels, L]
    h1 = F.relu(h1)  # ReLU after conv1

    # conv2 (no ReLU)
    h = F.conv1d(h1, w2, b2, padding=padding)   # [N, half_channels, L]

    # Apply mask: elementwise multiply by mask[i, 0, :]
    h = h * mask  # mask shape [N, 1, L], broadcast over channels

    # Update x1: forward x1 = x1 + h; reverse x1 = x1 - h
    if reverse:
        x1 = x1 - h
    else:
        x1 = x1 + h

    # Concatenate x0 and updated x1 back into x_full: shape [N, C, L]
    x_full = torch.empty((N, C, L), device=device, dtype=x_curr.dtype)
    grid_concat = (N, half, triton.cdiv(L, 128))
    concat_halves_forward[grid_concat](
        x0, x1, x_full,
        N, half, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        x_full.stride(0), x_full.stride(1), x_full.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Apply final mask to x_full: elementwise multiply by mask[i, 0, :]
    grid_mul = (N, x_full.shape[1], triton.cdiv(L, 128))
    mul_mask_kernel[grid_mul](
        x_full, mask,
        N, x_full.shape[1], L,
        x_full.stride(0), x_full.stride(1), x_full.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Return the updated concatenated tensor (x_full), which replaces the current x_curr
    return x_full


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool,
        transform_0_conv0_weight: torch.Tensor,
        transform_0_conv0_bias: torch.Tensor,
        transform_0_conv1_weight: torch.Tensor,
        transform_0_conv1_bias: torch.Tensor,
        transform_0_conv2_weight: torch.Tensor,
        transform_0_conv2_bias: torch.Tensor,
        transform_1_conv0_weight: torch.Tensor,
        transform_1_conv0_bias: torch.Tensor,
        transform_1_conv1_weight: torch.Tensor,
        transform_1_conv1_bias: torch.Tensor,
        transform_1_conv2_weight: torch.Tensor,
        transform_1_conv2_bias: torch.Tensor,
        transform_2_conv0_weight: torch.Tensor,
        transform_2_conv0_bias: torch.Tensor,
        transform_2_conv1_weight: torch.Tensor,
        transform_2_conv1_bias: torch.Tensor,
        transform_2_conv2_weight: torch.Tensor,
        transform_2_conv2_bias: torch.Tensor,
        transform_3_conv0_weight: torch.Tensor,
        transform_3_conv0_bias: torch.Tensor,
        transform_3_conv1_weight: torch.Tensor,
        transform_3_conv1_bias: torch.Tensor,
        transform_3_conv2_weight: torch.Tensor,
        transform_3_conv2_bias: torch.Tensor,
    ):
        device = x.device

        # Apply 4 transforms sequentially (forward or reverse), using Triton for coupling.
        x = single_transform_triton(
            x,
            x_mask,
            transform_0_conv0_weight, transform_0_conv0_bias,
            transform_0_conv1_weight, transform_0_conv1_bias,
            transform_0_conv2_weight, transform_0_conv2_bias,
            reverse,
            device,
        )

        x = single_transform_triton(
            x,
            x_mask,
            transform_1_conv0_weight, transform_1_conv0_bias,
            transform_1_conv1_weight, transform_1_conv1_bias,
            transform_1_conv2_weight, transform_1_conv2_bias,
            reverse,
            device,
        )

        x = single_transform_triton(
            x,
            x_mask,
            transform_2_conv0_weight, transform_2_conv0_bias,
            transform_2_conv1_weight, transform_2_conv1_bias,
            transform_2_conv2_weight, transform_2_conv2_bias,
            reverse,
            device,
        )

        x = single_transform_triton(
            x,
            x_mask,
            transform_3_conv0_weight, transform_3_conv0_bias,
            transform_3_conv1_weight, transform_3_conv1_bias,
            transform_3_conv2_weight, transform_3_conv2_bias,
            reverse,
            device,
        )

        return x


def run(*args):
    return ModelNew()(*args)
