import math
import torch
import torch.nn as nn
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
    # y2c: [N, 2*C_half, L]; read first C_half to y0, second C_half to y1
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
    ch = tl.program_id(1)  # channel index in [0, C_half)
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
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # elementwise ReLU: y = max(x, 0)
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L
    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask_out)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,  # mask shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    # elementwise multiply: y *= mask[:, 0, :]
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


def run_triton_only(
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
    """
    Triton-accelerated version of the residual coupling flow block.
    Numerical convolutions are done with PyTorch for correctness; Triton is used for split/concat/mul.
    """
    C = x.shape[1]
    half = C // 2
    N, C, L = x.shape

    # Helper to perform one transform using Triton split/concat, PyTorch conv+ReLU
    def do_transform(x_curr, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, reverse: bool):
        # Split halves
        x0 = torch.empty((N, half, L), device=x_curr.device, dtype=x_curr.dtype)
        x1 = torch.empty((N, half, L), device=x_curr.device, dtype=x_curr.dtype)
        grid_split = (N, half, triton.cdiv(L, 128))
        concat_halves_backward[grid_split](
            x_curr, x0, x1,
            N, half, L,
            x_curr.stride(0), x_curr.stride(1), x_curr.stride(2),
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv0 -> ReLU
        h0 = F.conv1d(x0, conv0_w, conv0_b, padding=conv0_w.shape[2] // 2)
        grid_relu0 = (N, conv0_w.shape[0], triton.cdiv(L, 128))
        relu_kernel[grid_relu0](
            h0, h0,
            N, conv0_w.shape[0], L,
            h0.stride(0), h0.stride(1), h0.stride(2),
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv1 -> ReLU
        h1 = F.conv1d(h0, conv1_w, conv1_b, padding=conv1_w.shape[2] // 2)
        grid_relu1 = (N, conv1_w.shape[0], triton.cdiv(L, 128))
        relu_kernel[grid_relu1](
            h1, h1,
            N, conv1_w.shape[0], L,
            h1.stride(0), h1.stride(1), h1.stride(2),
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv2 (no ReLU)
        h2 = F.conv1d(h1, conv2_w, conv2_b, padding=conv2_w.shape[2] // 2)

        # Update x1: if reverse, subtract; else add
        if reverse:
            x1 = x1 - h2
        else:
            x1 = x1 + h2

        # Concatenate [x0, x1] along channels
        x_full = torch.empty((N, C, L), device=x_curr.device, dtype=x_curr.dtype)
        grid_concat = (N, half, triton.cdiv(L, 128))
        concat_halves_forward[grid_concat](
            x0, x1, x_full,
            N, half, L,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x_full.stride(0), x_full.stride(1), x_full.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # Multiply by mask
        grid_mul = (N, C, triton.cdiv(L, 128))
        mul_mask_kernel[grid_mul](
            x_full, x_mask,
            N, C, L,
            x_full.stride(0), x_full.stride(1), x_full.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            BLOCK_L=128, num_warps=4
        )

        return x_full

    # Apply 4 transforms sequentially if not reverse; reverse order if reverse
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
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            x = do_transform(x, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, reverse=False)
    else:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x = do_transform(x, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, reverse=True)

    return x


# Entry point required by evaluator
class ModelNew(nn.Module):
    def forward(self, *args):
        return run_triton_only(*args)


def run(*args):
    return ModelNew()(*args)
