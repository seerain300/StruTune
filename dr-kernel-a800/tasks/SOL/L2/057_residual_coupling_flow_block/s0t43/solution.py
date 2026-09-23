import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    # Grid: (N, tiles over C_out, T_out)
    pid_n = tl.program_id(0)
    pid_cblk = tl.program_id(1)
    pid_t = tl.program_id(2)

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Loop over input channels and kernel taps
    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = pid_t - k  # output time index pid_t corresponds to t_out; t_in = t_out - k
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x[n, ci, t_in] for all co in this tile
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            x_ptrs = x_ptr + x_offsets + co_offsets * x_stride_c
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            # Load w[co, ci, k] for all co in this tile
            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals

    # Store output
    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


@triton.jit
def conv1d_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    # Same as conv1d_forward, then apply ReLU before store
    pid_n = tl.program_id(0)
    pid_cblk = tl.program_id(1)
    pid_t = tl.program_id(2)

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = pid_t - k
            in_bounds = (t_in >= 0) & (t_in < T_in)

            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            x_ptrs = x_ptr + x_offsets + co_offsets * x_stride_c
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


@triton.jit
def split_halves_kernel(
    x_ptr, x0_ptr, x1_ptr,
    N, C_half, T,
    x_stride_n, x_stride_c, x_stride_t,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
):
    # Grid: (N, C_half, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # First half
    x_offsets = pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t
    val = tl.load(x_ptr + x_offsets)
    tl.store(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t, val)

    # Second half (original channel index c' = c + C_half)
    x_offsets1 = pid_n * x_stride_n + (pid_c + C_half) * x_stride_c + pid_t * x_stride_t
    val = tl.load(x_ptr + x_offsets1)
    tl.store(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t, val)


@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True for forward (add), False for reverse (subtract)
    BLOCK_C: tl.constexpr,
):
    # Grid: (N, tiles over C, T_out)
    pid_n = tl.program_id(0)
    pid_cblk = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_start = pid_cblk * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    x1_vals = tl.load(x1_ptr + pid_n * x1_stride_n + c_offsets * x1_stride_c + pid_t * x1_stride_t, mask=c_mask, other=0.0)
    h_vals = tl.load(h_ptr + pid_n * h_stride_n + c_offsets * h_stride_c + pid_t * h_stride_t, mask=c_mask, other=0.0)
    if ADD:
        res = x1_vals + h_vals
    else:
        res = x1_vals - h_vals
    tl.store(out_ptr + pid_n * out_stride_n + c_offsets * out_stride_c + pid_t * out_stride_t, res, mask=c_mask)


@triton.jit
def cat_halves_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, T,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, C_half, T) to fill first half, and (N, C_half, T) for second half
    # We launch twice: for c in [0..C_half-1], write x0 and then x1 into out at channel c and c+C_half
    # To simplify, we can write directly: out[:, :C_half, :] = x0; out[:, C_half:, :] = x1
    # But Triton grid is 3D; we implement two launches here (host decides). In ModelNew.forward we'll launch twice.

    # First half
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)
    # Copy x0 to out[:, :C_half, :]
    x0_val = tl.load(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, x0_val)

    # Second half
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [0..C_half-1]
    pid_t = tl.program_id(2)
    x1_val = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    tl.store(out_ptr + pid_n * out_stride_n + (pid_c + C_half) * out_stride_c + pid_t * out_stride_t, x1_val)


@triton.jit
def mask_mul_kernel(
    h_ptr, mask_ptr, out_ptr,
    N, C, T,
    h_stride_n, h_stride_c, h_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,  # mask has shape [N, 1, T], but we use t dimension
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    # Grid: (N, tiles over C, T)
    pid_n = tl.program_id(0)
    pid_cblk = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_start = pid_cblk * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    h_vals = tl.load(h_ptr + pid_n * h_stride_n + c_offsets * h_stride_c + pid_t * h_stride_t, mask=c_mask, other=0.0)

    # mask is [N, 1, T] -> load mask[pid_n, 0, pid_t]
    mask_val = tl.load(mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + pid_t * mask_stride_t)
    out_vals = h_vals * mask_val

    tl.store(out_ptr + pid_n * out_stride_n + c_offsets * out_stride_c + pid_t * out_stride_t, out_vals, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool,
                # first transform weights
                transform_0_conv0_weight: torch.Tensor, transform_0_conv0_bias: torch.Tensor,
                transform_0_conv1_weight: torch.Tensor, transform_0_conv1_bias: torch.Tensor,
                transform_0_conv2_weight: torch.Tensor, transform_0_conv2_bias: torch.Tensor,
                # second transform weights
                transform_1_conv0_weight: torch.Tensor, transform_1_conv0_bias: torch.Tensor,
                transform_1_conv1_weight: torch.Tensor, transform_1_conv1_bias: torch.Tensor,
                transform_1_conv2_weight: torch.Tensor, transform_1_conv2_bias: torch.Tensor,
                # third transform weights
                transform_2_conv0_weight: torch.Tensor, transform_2_conv0_bias: torch.Tensor,
                transform_2_conv1_weight: torch.Tensor, transform_2_conv1_bias: torch.Tensor,
                transform_2_conv2_weight: torch.Tensor, transform_2_conv2_bias: torch.Tensor,
                # fourth transform weights
                transform_3_conv0_weight: torch.Tensor, transform_3_conv0_bias: torch.Tensor,
                transform_3_conv1_weight: torch.Tensor, transform_3_conv1_bias: torch.Tensor,
                transform_3_conv2_weight: torch.Tensor, transform_3_conv2_bias: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        x_mask = x_mask.contiguous().to(torch.float32)

        N, C, T = x.shape
        C_half = C // 2
        device = x.device
        x0 = x[:, :C_half, :]
        x1 = x[:, C_half:, :]

        # We need to loop over 4 transforms
        # We will apply each transform's convs and update x1 accordingly. We'll define helper functions to use Triton.

        # Helper: apply one transform using Triton convs + ReLU, returns h of shape [N, 96, T_out]
        def apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
            # conv0 forward: in=96, out=192, K=5
            C_in0 = x0.shape[1]  # 96
            C_out0 = conv0_w.shape[0]  # 192
            K0 = conv0_w.shape[2]  # 5
            T_in0 = x0.shape[2]
            T_out0 = T_in0 - K0 + 1

            y0 = torch.empty((N, C_out0, T_out0), device=device, dtype=torch.float32)
            grid0 = (N, triton.cdiv(C_out0, 64), T_out0)
            conv1d_forward_kernel[grid0](
                x0, conv0_w, conv0_b, y0,
                N, C_in0, T_in0, C_out0, T_out0, K0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_C=64,
            )

            # ReLU conv0 output
            y0_relu = torch.empty((N, C_out0, T_out0), device=device, dtype=torch.float32)
            grid0_relu = (N, triton.cdiv(C_out0, 64), T_out0)
            conv1d_relu_kernel[grid0_relu](
                y0, y0_relu,
                N, C_in0, T_in0, C_out0, T_out0, K0,
                y0.stride(0), y0.stride(1), y0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK_C=64,
            )

            # conv1 forward: in=192, out=192, K=5
            C_in1 = y0_relu.shape[1]  # 192
            C_out1 = conv1_w.shape[0]  # 192
            K1 = conv1_w.shape[2]  # 5
            T_in1 = y0_relu.shape[2]
            T_out1 = T_in1 - K1 + 1

            y1 = torch.empty((N, C_out1, T_out1), device=device, dtype=torch.float32)
            grid1 = (N, triton.cdiv(C_out1, 64), T_out1)
            conv1d_forward_kernel[grid1](
                y0_relu, conv1_w, conv1_b, y1,
                N, C_in1, T_in1, C_out1, T_out1, K1,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_C=64,
            )

            # ReLU conv1 output
            y1_relu = torch.empty((N, C_out1, T_out1), device=device, dtype=torch.float32)
            grid1_relu = (N, triton.cdiv(C_out1, 64), T_out1)
            conv1d_relu_kernel[grid1_relu](
                y1, y1_relu,
                N, C_in1, T_in1, C_out1, T_out1, K1,
                y1.stride(0), y1.stride(1), y1.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK_C=64,
            )

            # conv2 forward: in=192, out=96, K=5
            C_in2 = y1_relu.shape[1]  # 192
            C_out2 = conv2_w.shape[0]  # 96
            K2 = conv2_w.shape[2]  # 5
            T_in2 = y1_relu.shape[2]
            T_out2 = T_in2 - K2 + 1

            h = torch.empty((N, C_out2, T_out2), device=device, dtype=torch.float32)
            grid2 = (N, triton.cdiv(C_out2, 64), T_out2)
            conv1d_forward_kernel[grid2](
                y1_relu, conv2_w, conv2_b, h,
                N, C_in2, T_in2, C_out2, T_out2, K2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_C=64,
            )
            return h

        # Apply the 4 transforms
        # transform 0
        half_channels = C_half  # 96
        # We will keep x0, x1 updated in-place across transforms.
        for _ in range(4):
            # Compute h = apply_transform(x0) using Triton convs + ReLU
            h = apply_transform_triton(x0, transform_0_conv0_weight, transform_0_conv0_bias,
                                       transform_0_conv1_weight, transform_0_conv1_bias,
                                       transform_0_conv2_weight, transform_0_conv2_bias)

            # Multiply by mask (generic; mask is [N,1,T], apply per element)
            h_masked = torch.empty_like(h)
            grid_mask = (N, triton.cdiv(h.shape[1], 64), h.shape[2])
            mask_mul_kernel[grid_mask](
                h, x_mask, h_masked,
                N, h.shape[1], h.shape[2],
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK_C=64,
            )
            h = h_masked

            # Update x1: forward adds, reverse subtracts
            # We need to launch add_halves_kernel
            # Allocate out1 for updated x1
            out1 = torch.empty_like(x1)
            grid_add = (N, triton.cdiv(half_channels, 64), T)
            # Launch add for forward, subtract for reverse
            add_flag = 1 if not reverse else 0
            add_halves_kernel[grid_add](
                x1, h, out1,
                N, half_channels, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                out1.stride(0), out1.stride(1), out1.stride(2),
                ADD=add_flag,
                BLOCK_C=64,
            )

            # Swap x1 with out1 for next iteration
            x1 = out1

            # Advance to next transform: need to swap x0 as well. For simplicity, recompute x0 from x for each transform
            # (x0 is the first half of x at each step; in this model, x0 is the first 96 channels of the current x, which we already split).
            # However, since x is not modified by this function except x1, and the next transform uses the current x0, we need to
            # recompute x0 from the original x at the start of each transform. To avoid mutating x, we simply recompute x0 from x.
            # In this code, x0 is fixed as x[:, :C_half, :], and x1 is updated. We will just reuse x0 unchanged for each transform,
            # because original run does not depend on previous updates to x1 for subsequent transforms' x0.

            # For the next transform, we reuse x0 unchanged, and update x1 accordingly.

        # Finally, concatenate x0 and x1 back into x_out of shape [N, 2*C_half, T]
        # We will use Triton cat_halves_kernel to fill the output tensor
        x_out = torch.empty((N, 2 * C_half, T), device=device, dtype=torch.float32)
        grid_cat = (N, triton.cdiv(C_half, 64), T)
        cat_halves_kernel[grid_cat](
            x0, x1, x_out,
            N, C_half, T,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
        )

        return x_out


def run(*args):
    return ModelNew()(*args)
