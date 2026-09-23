import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv1d_triton(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, OC: tl.constexpr, T,
    x_sN, x_sC, x_sT,
    w_sO, w_sI, w_sK,
    out_sN, out_sC, out_sT,
    IC: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Triton implementation of Conv1d (cross-correlation) with stride=1 and padding=PAD.
    x: [N, IC, T] (float32)
    w: [OC, IC, K] (float32)
    b: [OC] (float32)
    out: [N, OC, T] (float32)

    Grid: (N, OC, ceil_div(T, BLOCK_T))
    Accumulate in float32, store with mask for valid t_out.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_out_start = pid_block * BLOCK_T
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)
    valid_t = t_out_idx < T

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Unrolled loops over input channels and kernel taps
    for ic in tl.static_range(IC):
        for k in tl.static_range(K):
            t_in = t_out_idx + k - PAD
            valid_t_in = valid_t & (t_in >= 0) & (t_in < T)
            # Load x[n, ic, t_in]
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_t_in, other=0.0)
            # Load w[oc, ic, k]
            w_val = tl.load(w_ptr + pid_oc * w_sO + ic * w_sI + k * w_sK)
            # Accumulate
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    # Store to out[n, oc, t_out_idx] with mask
    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr + out_offsets, acc, mask=valid_t)


@triton.jit
def multiply_mask_triton(
    y_ptr, mask_ptr, out_ptr,
    N, C, T,
    y_sN, y_sC, y_sT,
    mask_sN, mask_sC, mask_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiply: out[n, c, t] = y[n, c, t] * mask[n, 0, t]
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    y_offsets = pid_n * y_sN + pid_c * y_sC + t_idx * y_sT
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT

    y_vals = tl.load(y_ptr + y_offsets, mask=valid_t, other=0.0)
    m_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=1.0)

    out_vals = y_vals * m_vals

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def relu_triton(
    y_ptr, out_ptr,
    N, C, T,
    y_sN, y_sC, y_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Elementwise ReLU: out[n, c, t] = max(y[n, c, t], 0)
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    y_offsets = pid_n * y_sN + pid_c * y_sC + t_idx * y_sT
    y_vals = tl.load(y_ptr + y_offsets, mask=valid_t, other=0.0)

    # ReLU
    out_vals = tl.maximum(y_vals, 0.0)

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def add_h_triton(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_sN, x1_sC, x1_sT,
    h_sN, h_sC, h_sT,
    out_sN, out_sC, out_sT,
    ADD: tl.constexpr,  # bool constexpr: True for addition, False for subtraction
    BLOCK: tl.constexpr,
):
    """
    Elementwise affine coupling: out[n, c, t] = x1[n, c, t] + (+/-) h[n, c, t]
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    x1_offsets = pid_n * x1_sN + pid_c * x1_sC + t_idx * x1_sT
    h_offsets = pid_n * h_sN + pid_c * h_sC + t_idx * h_sT
    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT

    x1_vals = tl.load(x1_ptr + x1_offsets, mask=valid_t, other=0.0)
    h_vals = tl.load(h_ptr + h_offsets, mask=valid_t, other=0.0)

    if ADD:
        out_vals = x1_vals + h_vals
    else:
        out_vals = x1_vals - h_vals

    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def multiply_full_mask_triton(
    y_ptr, mask_ptr, out_ptr,
    N, C, T,
    y_sN, y_sC, y_sT,
    mask_sN, mask_sC, mask_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Apply full mask over all channels: out[n, c, t] = y[n, c, t] * mask[n, 0, t]
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    y_offsets = pid_n * y_sN + pid_c * y_sC + t_idx * y_sT
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT

    y_vals = tl.load(y_ptr + y_offsets, mask=valid_t, other=0.0)
    m_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=1.0)

    out_vals = y_vals * m_vals

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


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
    - Conv1d implemented via Triton kernel.
    - Mask application, ReLU, and affine coupling implemented via Triton kernels.
    - No torch.conv1d or torch.cat in the forward.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 for half_channels=96."
    half_channels = C // 2

    # Process 4 transforms in forward order; in reverse order when reverse=True
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

    # Process transforms sequentially; for reverse, iterate in reverse
    if not reverse:
        tforms = transforms
    else:
        tforms = list(reversed(transforms))

    for w0, b0, w1, b1, w2, b2 in tforms:
        # Split current x into x0 and x1
        x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
        x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

        # conv0: IC0=96, OC=192, K=5, padding=2
        IC0 = 96
        OC0 = 192
        K0 = 5
        PAD0 = K0 // 2
        y0 = torch.empty((N, OC0, T), dtype=x.dtype, device=x.device)
        grid_conv0 = (N, OC0, triton.cdiv(T, 128))
        conv1d_triton[grid_conv0](
            x0, w0, b0, y0,
            N, OC0, T,
            x0.stride(0), x0.stride(1), x0.stride(2),
            w0.stride(0), w0.stride(1), w0.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            IC=IC0, K=K0, PAD=PAD0, BLOCK_T=128, num_warps=4
        )

        # apply mask and ReLU to y0
        y0_masked = torch.empty_like(y0)
        multiply_full_mask_triton[(N, OC0, triton.cdiv(T, 128))](
            y0, x_mask, y0_masked,
            N, OC0, T,
            y0.stride(0), y0.stride(1), y0.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
            BLOCK=128, num_warps=4
        )
        y0_relu = torch.empty_like(y0_masked)
        relu_triton[(N, OC0, triton.cdiv(T, 128))](
            y0_masked, y0_relu,
            N, OC0, T,
            y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
            y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv1: IC1=192, OC=192, K=5, padding=2
        IC1 = 192
        OC1 = 192
        K1 = 5
        PAD1 = K1 // 2
        y1 = torch.empty((N, OC1, T), dtype=x.dtype, device=x.device)
        grid_conv1 = (N, OC1, triton.cdiv(T, 128))
        conv1d_triton[grid_conv1](
            y0_relu, w1, b1, y1,
            N, OC1, T,
            y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            w1.stride(0), w1.stride(1), w1.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            IC=IC1, K=K1, PAD=PAD1, BLOCK_T=128, num_warps=4
        )

        # apply mask and ReLU to y1
        y1_masked = torch.empty_like(y1)
        multiply_full_mask_triton[(N, OC1, triton.cdiv(T, 128))](
            y1, x_mask, y1_masked,
            N, OC1, T,
            y1.stride(0), y1.stride(1), y1.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
            BLOCK=128, num_warps=4
        )
        y1_relu = torch.empty_like(y1_masked)
        relu_triton[(N, OC1, triton.cdiv(T, 128))](
            y1_masked, y1_relu,
            N, OC1, T,
            y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv2: IC2=192, OC=96, K=5, padding=2
        IC2 = 192
        OC2 = 96
        K2 = 5
        PAD2 = K2 // 2
        h = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)
        grid_conv2 = (N, OC2, triton.cdiv(T, 128))
        conv1d_triton[grid_conv2](
            y1_relu, w2, b2, h,
            N, OC2, T,
            y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            IC=IC2, K=K2, PAD=PAD2, BLOCK_T=128, num_warps=4
        )

        # apply mask to h
        h_masked = torch.empty_like(h)
        multiply_full_mask_triton[(N, OC2, triton.cdiv(T, 128))](
            h, x_mask, h_masked,
            N, OC2, T,
            h.stride(0), h.stride(1), h.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # affine coupling: update x1 (second half)
        # If not reverse, add; if reverse, subtract
        out_x1 = torch.empty_like(x1)
        add_h_triton[(N, 96, triton.cdiv(T, 128))](
            x1, h_masked, out_x1,
            N, 96, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
            ADD=not reverse, BLOCK=128, num_warps=4
        )

        # Reconstruct x for next transform: out[:, :96, :] = y0_relu; out[:, 96:, :] = out_x1
        out = torch.empty((N, C, T), dtype=x.dtype, device=x.device)
        # write first half
        for oc in range(OC0):
            out[:, oc, :] = y0_relu[:, oc, :]
        # write second half
        for oc in range(OC2):
            out[:, OC0 + oc, :] = out_x1[:, oc, :]

        # Apply full mask to the reconstructed output (broadcast over channels)
        out_masked = torch.empty_like(out)
        multiply_full_mask_triton[(N, C, triton.cdiv(T, 128))](
            out, x_mask, out_masked,
            N, C, T,
            out.stride(0), out.stride(1), out.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            out_masked.stride(0), out_masked.stride(1), out_masked.stride(2),
            BLOCK=128, num_warps=4
        )

        # Continue with updated x
        x = out_masked

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # The original forward signature expects:
        # x, x_mask, reverse, and 12 weight tensors for 4 transforms
        # We simply call run with the provided args.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
