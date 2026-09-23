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

    # Store results for this block of time
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
    Elementwise add/subtract: out[n, c, t] = x1[n, c, t] + h[n, c, t] if not reverse, else -h
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
    Triton-only implementation of the original forward.
    - No torch.conv1d or torch.cat in the forward.
    - Conv1d, mask application, ReLU, and coupling updates are implemented via Triton kernels.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 (half_channels=96)."
    half_channels = C // 2
    BLOCK = 128

    # Precompute constants
    # We will process transforms sequentially; for each transform we:
    # 1) Split x into x0 and x1
    # 2) Compute conv0(x0) via Triton, apply mask and ReLU
    # 3) Compute conv1(result) via Triton, apply mask and ReLU
    # 4) Compute conv2(result) via Triton
    # 5) Apply mask, perform coupling on x1, construct new x by writing into final_out
    # 6) Apply mask to final_out

    # Helper to process one transform and update final_out
    def process_transform(
        w0, b0, w1, b1, w2, b2, is_add: bool
    ):
        # Split current x into halves
        x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
        x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

        # conv0: IC=96, OC=192, K=5, pad=2 -> out [N, 192, T]
        out0 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
        grid_conv0 = (N, 192, triton.cdiv(T, 128))
        conv1d_triton[grid_conv0](
            x0, w0, b0, out0,
            N, 96, 192, T, 5, 2,
            x0.stride(0), x0.stride(1), x0.stride(2),
            w0.stride(0), w0.stride(1), w0.stride(2),
            out0.stride(0), out0.stride(1), out0.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # apply mask and ReLU to out0
        out0_masked = torch.empty_like(out0)
        scale_mask_triton[(N, 192, triton.cdiv(T, 128))](
            out0, x_mask, out0_masked, N, 192, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            out0.stride(0), out0.stride(1), out0.stride(2),
            BLOCK=128, num_warps=4
        )
        out0_relu = torch.empty_like(out0_masked)
        relu_triton[(N, 192, triton.cdiv(T, 128))](
            out0_masked, out0_relu, N, 192, T,
            out0_masked.stride(0), out0_masked.stride(1), out0_masked.stride(2),
            out0_relu.stride(0), out0_relu.stride(1), out0_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv1: IC=192, OC=192, K=5, pad=2 -> out [N, 192, T]
        out1 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
        grid_conv1 = (N, 192, triton.cdiv(T, 128))
        conv1d_triton[grid_conv1](
            out0_relu, w1, b1, out1,
            N, 192, 192, T, 5, 2,
            out0_relu.stride(0), out0_relu.stride(1), out0_relu.stride(2),
            w1.stride(0), w1.stride(1), w1.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # apply mask and ReLU to out1
        out1_masked = torch.empty_like(out1)
        scale_mask_triton[(N, 192, triton.cdiv(T, 128))](
            out1, x_mask, out1_masked, N, 192, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            BLOCK=128, num_warps=4
        )
        out1_relu = torch.empty_like(out1_masked)
        relu_triton[(N, 192, triton.cdiv(T, 128))](
            out1_masked, out1_relu, N, 192, T,
            out1_masked.stride(0), out1_masked.stride(1), out1_masked.stride(2),
            out1_relu.stride(0), out1_relu.stride(1), out1_relu.stride(2),
            BLOCK=128, num_warps=4
        )

        # conv2: IC=192, OC=96, K=5, pad=2 -> h [N, 96, T]
        h = torch.empty((N, 96, T), dtype=x.dtype, device=x.device)
        grid_conv2 = (N, 96, triton.cdiv(T, 128))
        conv1d_triton[grid_conv2](
            out1_relu, w2, b2, h,
            N, 192, 96, T, 5, 2,
            out1_relu.stride(0), out1_relu.stride(1), out1_relu.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_T=128, num_warps=4
        )

        # apply mask to h (broadcast along channels)
        h_masked = torch.empty_like(h)
        scale_mask_triton[(N, 96, triton.cdiv(T, 128))](
            h, x_mask, h_masked, N, 96, T,
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK=128, num_warps=4
        )

        # Update x1 via coupling: x1 = x1 + h or x1 = x1 - h
        new_x1 = torch.empty_like(x1)
        add_h_triton[(N, 96, triton.cdiv(T, 128))](
            x1, h_masked, new_x1, N, 96, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            new_x1.stride(0), new_x1.stride(1), new_x1.stride(2),
            reverse_flag=reverse, BLOCK=128, num_warps=4
        )

        # Construct new x = [x0, new_x1]
        # We write into final_out and then set x to final_out at the end.
        # However, to avoid a global write here, we will just return (x0, new_x1) and handle final_out outside.

        return (x0, new_x1)

    # We need a final_out tensor to apply the last mask and assign back to x.
    final_out = torch.empty((N, C, T), dtype=x.dtype, device=x.device)

    # Process transforms in forward or reverse order
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
        # Forward: update x sequentially
        # We will perform each transform and write into final_out accordingly.
        # But since we don't have the intermediate x0 to inject into final_out, we build x step-by-step.
        # However, we need a unified final_out at the end. Simpler: perform each transform on current x,
        # but we don't have x0 to split; so we will instead reconstruct x step-by-step by updating
        # slices in the final_out buffer. To do that, we need the original x to split. Since the original
        # code relies on the current x at each transform, we need to keep the current x throughout.
        # So we will maintain a running final_out and update it after each transform.

        # Initialize final_out as the original x (unmasked)
        final_out.copy_(x)

        for w0, b0, w1, b1, w2, b2 in transforms:
            # Split current final_out into x0 and x1
            x0 = final_out[:, :half_channels, :].contiguous()
            x1 = final_out[:, half_channels:, :].contiguous()

            # Compute h as above
            _, new_x1 = process_transform(w0, b0, w1, b1, w2, b2, is_add=True)

            # Update final_out: first half unchanged (x0), second half updated
            final_out[:, :half_channels, :] = x0
            final_out[:, half_channels:, :] = new_x1

            # Apply mask to final_out (scale along time)
            scale_mask_triton[(N, C, triton.cdiv(T, 128))](
                final_out, x_mask, final_out, N, C, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                BLOCK=128, num_warps=4
            )

        # Assign final_out as the new x
        x = final_out
    else:
        # Reverse: iterate transforms in reversed order and subtract coupling
        final_out.copy_(x)

        for w0, b0, w1, b1, w2, b2 in reversed(transforms):
            # Split current final_out into x0 and x1
            x0 = final_out[:, :half_channels, :].contiguous()
            x1 = final_out[:, half_channels:, :].contiguous()

            _, new_x1 = process_transform(w0, b0, w1, b1, w2, b2, is_add=False)

            # Update final_out: first half unchanged, second half updated
            final_out[:, :half_channels, :] = x0
            final_out[:, half_channels:, :] = new_x1

            # Apply mask to final_out
            scale_mask_triton[(N, C, triton.cdiv(T, 128))](
                final_out, x_mask, final_out, N, C, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                final_out.stride(0), final_out.stride(1), final_out.stride(2),
                BLOCK=128, num_warps=4
            )

        x = final_out

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # args: x, x_mask, reverse, transform weights
        # Extract inputs
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects at least two arguments: x and x_mask.")
        x = args[0]
        x_mask = args[1]
        reverse = bool(args[2]) if len(args) > 2 else False

        # Gather transform weights
        if len(args) < 17:
            raise RuntimeError("ModelNew.forward expects 16 transform weight arguments for forward.")
        (
            t0c0w, t0c0b, t0c1w, t0c1b, t0c2w, t0c2b,
            t1c0w, t1c0b, t1c1w, t1c1b, t1c2w, t1c2b,
            t2c0w, t2c0b, t2c1w, t2c1b, t2c2w, t2c2b,
            t3c0w, t3c0b, t3c1w, t3c1b, t3c2w, t3c2b
        ) = args[3:]

        # Call run
        return run(x, x_mask, reverse,
                   t0c0w, t0c0b, t0c1w, t0c1b, t0c2w, t0c2b,
                   t1c0w, t1c0b, t1c1w, t1c1b, t1c2w, t1c2b,
                   t2c0w, t2c0b, t2c1w, t2c1b, t2c2w, t2c2b,
                   t3c0w, t3c0b, t3c1w, t3c1b, t3c2w, t3c2b)


def run(*args):
    return ModelNew()(*args)
