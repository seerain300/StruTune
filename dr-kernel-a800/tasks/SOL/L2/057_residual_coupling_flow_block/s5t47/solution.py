import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def relu_inplace_triton(out_ptr, B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    In-place ReLU over a 3D tensor [B, C, T].
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    out_index = (pid_b * C + pid_c) * T + t_offsets
    val = tl.load(out_ptr + out_index, mask=mask_t, other=0.0)
    val = tl.maximum(val, 0.0)  # ReLU
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def apply_mask_to_h_triton(h_ptr, mask_ptr, h_out_ptr, B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    Elementwise h_out = h * mask, where mask is [B, 1, T] (broadcast along channels).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    h_index = (pid_b * C + pid_c) * T + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    # mask_ptr is [B, 1, T]
    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    h_val = h_val * mask_val
    tl.store(h_out_ptr + h_index, h_val, mask=mask_t)


@triton.jit
def add_h_to_x1_triton(h_ptr, x1_ptr, x1_out_ptr, B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, ADD: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    In-place update: x1_out = x1 + h if ADD=1 else x1_out = x1 - h.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    index = (pid_b * C + pid_c) * T + t_offsets

    x1_val = tl.load(x1_ptr + index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + index, mask=mask_t, other=0.0)

    if ADD == 1:
        val = x1_val + h_val
    else:
        val = x1_val - h_val

    tl.store(x1_out_ptr + index, val, mask=mask_t)


@triton.jit
def concat_copy_first_half(x0_ptr, out_ptr, B: tl.constexpr, C0: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    Copy x0[b, :C0, :] into out[b, :C0, :].
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # pid_c in [0, C0)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    in_index = (pid_b * C0 + pid_c) * T + t_offsets
    out_index = (pid_b * C0 + pid_c) * T + t_offsets  # same region

    val = tl.load(x0_ptr + in_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def concat_copy_second_half(x1_ptr, out_ptr, B: tl.constexpr, C1: tl.constexpr, C0: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    """
    Copy x1[b, :C1, :] into out[b, C0:C0+C1, :].
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # pid_c in [0, C1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    in_index = (pid_b * C1 + pid_c) * T + t_offsets
    out_index = ((pid_b * (C0 + C1)) + (pid_c + C0)) * T + t_offsets

    val = tl.load(x1_ptr + in_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


def _pick_block_t(T):
    if T >= 4096:
        return 256
    elif T >= 2048:
        return 128
    elif T >= 1024:
        return 128
    else:
        return 64


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # transform weights
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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    # Ensure tensors on CUDA and contiguous
    device = x.device
    if not x.is_cuda:
        x = x.cuda(non_blocking=True)
    x = x.contiguous().to(torch.float32)
    x_mask = x_mask.contiguous().to(torch.float32)

    B, C, T = x.shape
    half_channels = C // 2

    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in [
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
        ]:
            # Split into two halves
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # conv0: hidden_channels out, half_channels in, padding=2 (for K=5)
            h = F.conv1d(x0, conv0_w, conv0_b, padding=2)
            # ReLU via Triton kernel
            h_relu = torch.empty_like(h)
            BLOCK_T = _pick_block_t(T)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            relu_inplace_triton[grid](h, B=B, C=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # apply mask h = h * x_mask (broadcast along channels)
            h_masked = torch.empty_like(h)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            apply_mask_to_h_triton[grid](h, x_mask, h_masked, B=B, C=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # Affine coupling: x1 = x1 + h
            x1_out = torch.empty_like(x1)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            add_h_to_x1_triton[grid](h_masked, x1, x1_out, B=B, C=half_channels, T=T, ADD=1, BLOCK_T=BLOCK_T, num_warps=4)

            # Concatenate back into output of shape [B, C, T]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            grid_first = (B, half_channels, triton.cdiv(T, BLOCK_T))
            concat_copy_first_half[grid_first](x0, out, B=B, C0=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)
            grid_second = (B, half_channels, triton.cdiv(T, BLOCK_T))
            concat_copy_second_half[grid_second](x1_out, out, B=B, C1=half_channels, C0=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # apply final x_mask across channels (no-op here since x_mask is ones)
            out_masked = torch.empty_like(out)
            grid = (B, C, triton.cdiv(T, BLOCK_T))
            apply_mask_to_h_triton[grid](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # update x for next iteration
            x = out_masked

    else:
        # Reverse pass: apply transformations in reverse order, subtract h
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed([
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
        ]):
            # Split into two halves
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # conv0: hidden_channels out, half_channels in, padding=2
            h = F.conv1d(x0, conv0_w, conv0_b, padding=2)
            # ReLU via Triton kernel
            h_relu = torch.empty_like(h)
            BLOCK_T = _pick_block_t(T)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            relu_inplace_triton[grid](h, B=B, C=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # apply mask h = h * x_mask (broadcast along channels)
            h_masked = torch.empty_like(h)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            apply_mask_to_h_triton[grid](h, x_mask, h_masked, B=B, C=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # Affine coupling: x1 = x1 - h
            x1_out = torch.empty_like(x1)
            grid = (B, half_channels, triton.cdiv(T, BLOCK_T))
            add_h_to_x1_triton[grid](h_masked, x1, x1_out, B=B, C=half_channels, T=T, ADD=0, BLOCK_T=BLOCK_T, num_warps=4)

            # Concatenate back into output of shape [B, C, T]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            grid_first = (B, half_channels, triton.cdiv(T, BLOCK_T))
            concat_copy_first_half[grid_first](x0, out, B=B, C0=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)
            grid_second = (B, half_channels, triton.cdiv(T, BLOCK_T))
            concat_copy_second_half[grid_second](x1_out, out, B=B, C1=half_channels, C0=half_channels, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # apply final x_mask across channels (no-op here since x_mask is ones)
            out_masked = torch.empty_like(out)
            grid = (B, C, triton.cdiv(T, BLOCK_T))
            apply_mask_to_h_triton[grid](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=BLOCK_T, num_warps=4)

            # update x for next iteration
            x = out_masked

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # args include: x, x_mask, reverse, and the 24 weights/bias tensors
        return run(*args)


def run(*args):
    return ModelNew()(*args)
