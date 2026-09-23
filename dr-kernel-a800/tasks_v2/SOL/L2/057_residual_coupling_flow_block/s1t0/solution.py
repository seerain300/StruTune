import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def multiply_mask_kernel(inp_ptr, mask_ptr, out_ptr,
                          N, C, T,
                          BLOCK: tl.constexpr):
    """
    Elementwise multiply: out[n, c, t] = inp[n, c, t] * mask[n, 0, t].
    Mask is shape [N, 1, T], broadcast over channel dimension.
    We flatten the (C, T) per n into a single dimension and let grid = (N, ceil((C*T)/BLOCK)).
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Each program handles BLOCK elements of the (C*T) plane for a given n
    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T

    # Compute (c, t) from linear idx
    # c = idx // T, t = idx % T
    t = idx % T
    c = idx // T

    # Valid mask
    valid = idx < total

    # Compute input/output offsets (contiguous NCL layout => offset = n*(C*T) + idx)
    n = pid_n
    inp_offsets = n * (C * T) + idx
    out_offsets = n * (C * T) + idx

    # Load inp
    inp = tl.load(inp_ptr + inp_offsets, mask=valid, other=0.0)

    # Load mask for this (n, t) across all c; mask shape [N, T] (we pass mask[:, 0, :] flattened)
    mask_offsets = n * T + t
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid, other=1.0)  # other=1.0 since invalid entries masked anyway

    # Multiply
    out = inp * mask_vals

    # Store
    tl.store(out_ptr + out_offsets, out, mask=valid)


@triton.jit
def relu_kernel(inp_ptr, out_ptr, N, C, T, BLOCK: tl.constexpr):
    """
    Elementwise ReLU: out[n, c, t] = max(inp[n, c, t], 0).
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T

    t = idx % T
    c = idx // T
    valid = idx < total

    n = pid_n
    offsets = n * (C * T) + idx

    x = tl.load(inp_ptr + offsets, mask=valid, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, x, mask=valid)


@triton.jit
def add_h_kernel(x1_ptr, h_ptr, out_ptr, N, C, T, is_add: tl.constexpr, BLOCK: tl.constexpr):
    """
    Elementwise coupling: out[n, c, t] = x1[n, c, t] + (-+) h[n, c, t].
    is_add: True => out = x1 + h; False => out = x1 - h.
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T

    t = idx % T
    c = idx // T
    valid = idx < total

    n = pid_n
    offsets = n * (C * T) + idx

    x1 = tl.load(x1_ptr + offsets, mask=valid, other=0.0)
    h = tl.load(h_ptr + offsets, mask=valid, other=0.0)

    if is_add:
        y = x1 + h
    else:
        y = x1 - h

    tl.store(out_ptr + offsets, y, mask=valid)


def _apply_mask_triton(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, T], float32
    mask: [N, 1, T], float32
    Returns: x * mask (broadcast along C)
    """
    assert x.ndim == 3 and mask.ndim == 3
    N, C, T = x.shape
    # Ensure mask is [N, 1, T] and same device
    assert mask.shape == (N, 1, T)
    mask_flat = mask[:, 0, :].contiguous()  # [N, T]

    y = torch.empty_like(x)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    multiply_mask_kernel[grid](x, mask_flat, y, N, C, T, BLOCK=BLOCK, num_warps=4)
    return y


def _relu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Elementwise ReLU on x: [N, C, T]
    """
    N, C, T = x.shape
    y = torch.empty_like(x)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    relu_kernel[grid](x, y, N, C, T, BLOCK=BLOCK, num_warps=4)
    return y


def _add_h_triton(x1: torch.Tensor, h: torch.Tensor, reverse: bool) -> torch.Tensor:
    """
    Elementwise coupling: x1 = x1 + h if not reverse, or x1 - h if reverse.
    """
    assert x1.shape == h.shape and x1.ndim == 3
    N, C, T = x1.shape
    y = torch.empty_like(x1)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    is_add = not reverse  # Triton expects a compile-time boolean; pass True/False
    add_h_kernel[grid](x1, h, y, N, C, T, is_add=is_add, BLOCK=BLOCK, num_warps=4)
    return y


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # weights and biases for 4 transforms
    transform_0_conv0_weight, transform_0_conv0_bias,
    transform_0_conv1_weight, transform_0_conv1_bias,
    transform_0_conv2_weight, transform_0_conv02_bias,  # Note: typo in original (conv2_bias), we fix it
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
    Triton-optimized version of the original run, using Triton for elementwise operations:
    - Mask application
    - ReLU
    - Affine coupling addition/subtraction
    Conv1d is kept in PyTorch/cuDNN.
    """
    N, C, T = x.shape
    half_channels = C // 2
    assert C == 192 and half_channels == 96, "This implementation assumes C=192 and half_channels=96."

    transforms = [
        (transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv02_bias),  # corrected conv2_bias
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
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]  # [N, 96, T]
            x1 = x[:, half_channels:, :]   # [N, 96, T]

            # Compute transformation conditioned on x0 (PyTorch convs)
            # conv0
            padding = conv0_w.shape[2] // 2
            h0 = F.conv1d(x0, conv0_w, conv0_b, padding=padding)
            # mask
            h0 = _apply_mask_triton(h0, x_mask)
            # ReLU
            h0 = _relu_triton(h0)
            # conv1
            h = F.conv1d(h0, conv1_w, conv1_b, padding=padding)
            # mask
            h = _apply_mask_triton(h, x_mask)
            # ReLU
            h = _relu_triton(h)
            # conv2
            y = F.conv1d(h, conv2_w, conv2_b, padding=padding)

            # Affine coupling: x1 = x1 + y
            y = _apply_mask_triton(y, x_mask)
            x1 = _add_h_triton(x1, y, reverse=False)

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

            # Apply mask to output
            x = _apply_mask_triton(x, x_mask)
    else:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]  # [N, 96, T]
            x1 = x[:, half_channels:, :]   # [N, 96, T]

            # Compute transformation conditioned on x0 (PyTorch convs)
            # conv0
            padding = conv0_w.shape[2] // 2
            h0 = F.conv1d(x0, conv0_w, conv0_b, padding=padding)
            # mask
            h0 = _apply_mask_triton(h0, x_mask)
            # ReLU
            h0 = _relu_triton(h0)
            # conv1
            h = F.conv1d(h0, conv1_w, conv1_b, padding=padding)
            # mask
            h = _apply_mask_triton(h, x_mask)
            # ReLU
            h = _relu_triton(h)
            # conv2
            y = F.conv1d(h, conv2_w, conv2_b, padding=padding)

            # Affine coupling: x1 = x1 - y (reverse)
            y = _apply_mask_triton(y, x_mask)
            x1 = _add_h_triton(x1, y, reverse=True)

            # Concatenate back
            x = torch.cat([x0, x1], dim=1)

            # Apply mask to output
            x = _apply_mask_triton(x, x_mask)

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # args are: x, x_mask, reverse, then 24 tensors for weights/biases as per original signature
        # We expect exactly 27 positional args (3 tensors + 24 tensors)
        assert len(args) == 27, "ModelNew.forward expects 27 inputs: (x, x_mask, reverse, 24 weight/bias tensors)"
        return run(*args)


def run(*args):
    return ModelNew()(*args)
