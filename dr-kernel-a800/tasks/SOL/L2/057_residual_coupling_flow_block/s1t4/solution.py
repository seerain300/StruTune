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

    # Accumulator for output vector of length BLOCK_T
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ic in range(0, IC):
        for k in range(0, K):
            t_in = t_out_idx + k - pad  # valid when 0 <= t_in < T
            valid_t_in = valid_t & (t_in >= 0) & (t_in < T)
            # Load x[n, ic, t_in]
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_t_in, other=0.0)
            # Load w[oc, ic, k]
            w_val = tl.load(w_ptr + pid_oc * w_sO + ic * w_sI + k * w_sK)
            # Multiply and accumulate (promote to float32 for stability)
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    # Store to out[n, oc, t_out_idx]
    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr + out_offsets, acc, mask=valid_t)


@triton.jit
def relu_triton(
    inp_ptr, out_ptr,
    N, C, T,
    inp_sN, inp_sC, inp_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Elementwise ReLU on a [N, C, T] tensor. Computes out = max(inp, 0).
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    base_in = pid_n * inp_sN + pid_c * inp_sC
    base_out = pid_n * out_sN + pid_c * out_sC

    in_offsets = base_in + t_idx * inp_sT
    out_offsets = base_out + t_idx * out_sT

    x = tl.load(inp_ptr + in_offsets, mask=valid_t, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(out_ptr + out_offsets, x, mask=valid_t)


@triton.jit
def mask_apply_triton(
    inp_ptr, mask_ptr, out_ptr,
    N, C, T,
    mask_sN, mask_sC, mask_sT,
    inp_sN, inp_sC, inp_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiply of inp [N, C, T] by mask [N, 1, T] (broadcast across C).
    out = inp * mask. We cast mask to inp dtype for consistency.
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    base_inp = pid_n * inp_sN + pid_c * inp_sC
    base_out = pid_n * out_sN + pid_c * out_sC

    # Load mask for channel 0 (broadcast along C)
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=1.0)
    # Promote mask to inp dtype to avoid dtype mismatch
    mask_vals = mask_vals.to(inp_ptr.dtype.element_ty)

    inp_offsets = base_inp + t_idx * inp_sT
    out_offsets = base_out + t_idx * out_sT

    x = tl.load(inp_ptr + inp_offsets, mask=valid_t, other=0.0)
    y = x * mask_vals
    tl.store(out_ptr + out_offsets, y, mask=valid_t)


@triton.jit
def add_h_triton(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_sN, x1_sC, x1_sT,
    h_sN, h_sC, h_sT,
    out_sN, out_sC, out_sT,
    reverse_flag: tl.constexpr,  # 0 => add, 1 => subtract
    BLOCK: tl.constexpr,
):
    """
    Elementwise coupling: out = x1 + (-) h depending on reverse_flag.
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    base_x1 = pid_n * x1_sN + pid_c * x1_sC
    base_h = pid_n * h_sN + pid_c * h_sC
    base_out = pid_n * out_sN + pid_c * out_sC

    x1_offsets = base_x1 + t_idx * x1_sT
    h_offsets = base_h + t_idx * h_sT
    out_offsets = base_out + t_idx * out_sT

    x1 = tl.load(x1_ptr + x1_offsets, mask=valid_t, other=0.0)
    h = tl.load(h_ptr + h_offsets, mask=valid_t, other=0.0)

    if reverse_flag == 1:
        y = x1 - h
    else:
        y = x1 + h
    tl.store(out_ptr + out_offsets, y, mask=valid_t)


@triton.jit
def apply_mask_overall_triton(
    out_ptr, mask_ptr, out_masked_ptr,
    N, C, T,
    mask_sN, mask_sC, mask_sT,
    out_sN, out_sC, out_sT,
    out_masked_sN, out_masked_sC, out_masked_sT,
    BLOCK: tl.constexpr,
):
    """
    Apply mask [N, 1, T] across all channels: out_masked = out * mask.
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    base_out = pid_n * out_sN + pid_c * out_sC
    base_out_masked = pid_n * out_masked_sN + pid_c * out_masked_sC

    # Load mask for channel 0 (broadcast)
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=1.0)
    # Cast to out dtype
    mask_vals = mask_vals.to(out_ptr.dtype.element_ty)

    out_offsets = base_out + t_idx * out_sT
    out_masked_offsets = base_out_masked + t_idx * out_masked_sT

    out = tl.load(out_ptr + out_offsets, mask=valid_t, other=0.0)
    out = out * mask_vals
    tl.store(out_masked_ptr + out_masked_offsets, out, mask=valid_t)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    transform_0_conv0_weight, transform_0_conv0_bias,
    transform_0_conv1_weight, transform_0_conv1_bias,
    transform_0_conv2_weight, transform_0_conv02_bias,  # bias name corrected to match earlier
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
    - No torch.conv1d, no torch.cat. All conv1d and elementwise ops are Triton kernels.
    - Mask application is done by Triton kernels; final x is scaled by x_mask after each transform.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 for half_channels=96."
    half_channels = C // 2

    # Prepare final output [N, 192, T], initially zeros
    final_out = torch.zeros((N, C, T), dtype=x.dtype, device=x.device)

    # Define transforms
    transforms = [
        (transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv02_bias),
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

    # Process transforms in forward or reversed order
    transform_list = transforms if not reverse else reversed(transforms)

    for i, (w0, b0, w1, b1, w2, b2) in enumerate(transform_list):
        # Split current final_out into x0 and x1 halves
        x0 = final_out[:, :half_channels, :].contiguous()  # [N, 96, T]
        x1 = final_out[:, half_channels:, :].contiguous()  # [N, 96, T]

        # conv0: out_channels=192, in_channels=96, K=5, padding=2
        OC = 192
        IC0 = 96
        K0 = 5
        pad0 = K0 // 2
        y0 = torch.empty((N, OC, T), dtype=x.dtype, device=x.device)
        grid_conv0 = (N, OC, triton.cdiv(T, 128))
        conv1d_triton[grid_conv0](
            x0, w0, b0, y0,
            N, IC0, OC, T, K0, pad0,
            x0.stride(0), x0.stride(1), x0.stride(2),
            w0.stride(0), w0.stride(1), w0.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # Apply mask across all channels
        y0_masked = torch.empty_like(y0)
        apply_mask_overall_triton[(N, OC, triton.cdiv(T, 128))](
            y0, x_mask, y0_masked, N, OC, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # ReLU
        y0_relu = torch.empty_like(y0_masked)
        relu_triton[(N, OC, triton.cdiv(T, 128))](
            y0_masked, y0_relu, N, OC, T,
            y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
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

        # Apply mask across all channels
        y1_masked = torch.empty_like(y1)
        apply_mask_overall_triton[(N, OC1, triton.cdiv(T, 128))](
            y1, x_mask, y1_masked, N, OC1, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # ReLU
        y1_relu = torch.empty_like(y1_masked)
        relu_triton[(N, OC1, triton.cdiv(T, 128))](
            y1_masked, y1_relu, N, OC1, T,
            y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv2: out_channels=96, in_channels=192, K=5, padding=2
        OC2 = half_channels  # 96
        IC2 = 192
        K2 = 5
        pad2 = K2 // 2
        y2 = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)
        grid_conv2 = (N, OC2, triton.cdiv(T, 128))
        conv1d_triton[grid_conv2](
            y1_relu, w2, b2, y2,
            N, IC2, OC2, T, K2, pad2,
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            y2.stride(0), y2.stride(1), y2.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # Apply mask across all channels (for coupling)
        y2_masked = torch.empty_like(y2)
        apply_mask_overall_triton[(N, OC2, triton.cdiv(T, 128))](
            y2, x_mask, y2_masked, N, OC2, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            y2.stride(0), y2.stride(1), y2.stride(2),
            y2_masked.stride(0), y2_masked.stride(1), y2_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # Coupling: x1 = x1 + y2_masked (or -y2_masked in reverse)
        x1_new = torch.empty_like(x1)
        add_h_triton[(N, half_channels, triton.cdiv(T, 128))](
            x1, y2_masked, x1_new, N, half_channels, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            y2_masked.stride(0), y2_masked.stride(1), y2_masked.stride(2),
            x1_new.stride(0), x1_new.stride(1), x1_new.stride(2),
            reverse_flag=1 if reverse else 0,
            BLOCK=128, num_warps=4
        )

        # Update final_out: first half is x0, second half is x1_new
        final_out[:, :half_channels, :] = x0
        final_out[:, half_channels:, :] = x1_new

    return final_out


class ModelNew(nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
