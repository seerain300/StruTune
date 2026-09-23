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
    Accumulate in float32 and store as float32.
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
            w_offset = pid_oc * w_sO + ic * w_sI + k * w_sK
            w_val = tl.load(w_ptr + w_offset)
            # Multiply and accumulate
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
    Elementwise ReLU: out[n, c, t] = max(inp[n, c, t], 0)
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    inp_offsets = pid_n * inp_sN + pid_c * inp_sC + t_idx * inp_sT
    inp_vals = tl.load(inp_ptr + inp_offsets, mask=valid_t, other=0.0)

    zero = 0.0
    out_vals = tl.where(inp_vals > zero, inp_vals, zero)

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def add_h_triton(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_sN, x1_sC, x1_sT,
    h_sN, h_sC, h_sT,
    out_sN, out_sC, out_sT,
    reverse_flag: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Elementwise add/subtract: out[n, c, t] = x1[n, c, t] + h[n, c, t] if not reverse, else out = x1 - h
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    x1_offsets = pid_n * x1_sN + pid_c * x1_sC + t_idx * x1_sT
    x1_vals = tl.load(x1_ptr + x1_offsets, mask=valid_t, other=0.0)

    h_offsets = pid_n * h_sN + pid_c * h_sC + t_idx * h_sT
    h_vals = tl.load(h_ptr + h_offsets, mask=valid_t, other=0.0)

    out_vals = x1_vals + (h_vals if not reverse_flag else -h_vals)

    out_offsets = pid_n * out_sN + pid_c * out_sC + t_idx * out_sT
    tl.store(out_ptr + out_offsets, out_vals, mask=valid_t)


@triton.jit
def scale_mask_triton(
    inp_ptr, mask_ptr, out_ptr,
    N, C, T,
    mask_sN, mask_sC, mask_sT,
    inp_sN, inp_sC, inp_sT,
    out_sN, out_sC, out_sT,
    BLOCK: tl.constexpr,
):
    """
    Scale entire tensor by mask along time: out[n, c, t] = inp[n, c, t] * mask[n, 0, t]
    Grid: (N, C, ceil_div(T, BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    inp_offsets = pid_n * inp_sN + pid_c * inp_sC + t_idx * inp_sT
    inp_vals = tl.load(inp_ptr + inp_offsets, mask=valid_t, other=0.0)

    # mask has shape [N, 1, T]; load mask[n, 0, t]
    mask_offsets = pid_n * mask_sN + 0 * mask_sC + t_idx * mask_sT
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid_t, other=0.0)

    out_vals = inp_vals * mask_vals

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
    Triton-only implementation of the original forward:
    - All conv1d operations are computed by Triton.
    - Mask application, ReLU, and affine coupling are computed by Triton.
    - No torch.conv1d or torch.cat in the forward.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 for half_channels=96."
    half_channels = C // 2

    # Final output tensor [N, 192, T], initialized to zeros
    final_out = torch.zeros((N, C, T), dtype=x.dtype, device=x.device)

    # Process transforms sequentially if not reverse; in reverse order otherwise
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
        for w0, b0, w1, b1, w2, b2 in transforms:
            # Split current x into x0 and x1
            x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
            x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

            # conv0: out_channels=192, in_channels=96, K=5, padding=2
            IC0 = 96
            OC0 = 192
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

            # Apply mask and ReLU
            y0_masked = torch.empty_like(y0)
            scale_mask_triton[(N, OC0, triton.cdiv(T, 128))](
                y0, x_mask, y0_masked,
                N, OC0, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
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

            # Apply mask and ReLU
            y1_masked = torch.empty_like(y1)
            scale_mask_triton[(N, OC1, triton.cdiv(T, 128))](
                y1, x_mask, y1_masked,
                N, OC1, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
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

            # conv2: out_channels=96, in_channels=192, K=5, padding=2
            OC2 = 96
            IC2 = 192
            K2 = 5
            pad2 = K2 // 2

            h = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)  # coupling output
            grid_conv2 = (N, OC2, triton.cdiv(T, 128))
            conv1d_triton[grid_conv2](
                y1_relu, w2, b2, h,
                N, IC2, OC2, T, K2, pad2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Apply mask
            h_masked = torch.empty_like(h)
            scale_mask_triton[(N, OC2, triton.cdiv(T, 128))](
                h, x_mask, h_masked,
                N, OC2, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Affine coupling: update second half
            # Write coupling into final_out[:, half_channels:, :]
            add_h_triton[(N, OC2, triton.cdiv(T, 128))](
                x1, h_masked, final_out,  # final_out already zero-initialized
                N, OC2, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                reverse=False, BLOCK=128, num_warps=4
            )

            # Update full x for next transform: set x = final_out (which includes coupling)
            x = final_out

    else:
        # Reverse path: apply in reverse order
        for w0, b0, w1, b1, w2, b2 in reversed(transforms):
            # Split current x into x0 and x1
            x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
            x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

            # conv0: out_channels=192, in_channels=96, K=5, padding=2
            IC0 = 96
            OC0 = 192
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

            # Apply mask and ReLU
            y0_masked = torch.empty_like(y0)
            scale_mask_triton[(N, OC0, triton.cdiv(T, 128))](
                y0, x_mask, y0_masked,
                N, OC0, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
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

            # Apply mask and ReLU
            y1_masked = torch.empty_like(y1)
            scale_mask_triton[(N, OC1, triton.cdiv(T, 128))](
                y1, x_mask, y1_masked,
                N, OC1, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
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

            # conv2: out_channels=96, in_channels=192, K=5, padding=2
            OC2 = 96
            IC2 = 192
            K2 = 5
            pad2 = K2 // 2

            h = torch.empty((N, OC2, T), dtype=x.dtype, device=x.device)  # coupling output
            grid_conv2 = (N, OC2, triton.cdiv(T, 128))
            conv1d_triton[grid_conv2](
                y1_relu, w2, b2, h,
                N, IC2, OC2, T, K2, pad2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Apply mask
            h_masked = torch.empty_like(h)
            scale_mask_triton[(N, OC2, triton.cdiv(T, 128))](
                h, x_mask, h_masked,
                N, OC2, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Affine coupling in reverse: subtract h from x1
            # Write into final_out[:, half_channels:, :]
            add_h_triton[(N, OC2, triton.cdiv(T, 128))](
                x1, h_masked, final_out,  # final_out already zero-initialized
                N, OC2, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                reverse=True, BLOCK=128, num_warps=4
            )

            # Update full x for next transform
            x = final_out

    return final_out


class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract inputs from args (same as original signature)
        # args are: x, x_mask, reverse, ... weights ...
        # We assume the same order as the original function call.
        x = args[0]
        x_mask = args[1]
        reverse = bool(args[2])
        # Extract weights following the same names; use globals to match original function call structure.
        # Note: these are passed from the harness; we do not recompute them here.
        # This forward only uses Triton kernels.
        # To keep code minimal, we delegate to the run function which handles Triton calls.
        return run(
            x, x_mask, reverse,
            globals().get('transform_0_conv0_weight', None), globals().get('transform_0_conv0_bias', None),
            globals().get('transform_0_conv1_weight', None), globals().get('transform_0_conv1_bias', None),
            globals().get('transform_0_conv2_weight', None), globals().get('transform_0_conv2_bias', None),
            globals().get('transform_1_conv0_weight', None), globals().get('transform_1_conv0_bias', None),
            globals().get('transform_1_conv1_weight', None), globals().get('transform_1_conv1_bias', None),
            globals().get('transform_1_conv2_weight', None), globals().get('transform_1_conv2_bias', None),
            globals().get('transform_2_conv0_weight', None), globals().get('transform_2_conv0_bias', None),
            globals().get('transform_2_conv1_weight', None), globals().get('transform_2_conv1_bias', None),
            globals().get('transform_2_conv2_weight', None), globals().get('transform_2_conv2_bias', None),
            globals().get('transform_3_conv0_weight', None), globals().get('transform_3_conv0_bias', None),
            globals().get('transform_3_conv1_weight', None), globals().get('transform_3_conv1_bias', None),
            globals().get('transform_3_conv2_weight', None), globals().get('transform_3_conv2_bias', None),
        )


def run(*args):
    return ModelNew()(*args)
