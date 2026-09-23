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
    # Grid: (N, ceil_div(C_out, BLOCK_C), T_out)
    pid_n = tl.program_id(0)
    pid_cblk = tl.program_id(1)
    pid_t = tl.program_id(2)

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    # Accumulator for this block of output channels
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Loop over input channels and kernel taps
    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            # For padding=0 (default in original code), T_out = T_in - K + 1.
            # Hence t = pid_t in [0, T_out-1] ensures t + k < T_in.
            t_in = pid_t + k  # since k starts from 0 and pid_t is output time index
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x[n, ci, t_in] for all co in block
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            co_vec_offsets = co_offsets * x_stride_c
            x_ptrs = x_ptr + x_offsets + co_vec_offsets
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            # Load w[co, ci, k] for all co in block
            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            # Accumulate
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
    in_ptr, out_ptr,
    N, C, T,
    in_stride_n, in_stride_c, in_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Elementwise ReLU over in_ptr -> out_ptr
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    val = tl.load(in_ptr + pid_n * in_stride_n + pid_c * in_stride_c + pid_t * in_stride_t)
    val = tl.maximum(val, 0.0)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, val)


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

    # First half (channel index pid_c)
    val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t)
    tl.store(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t, val)

    # Second half (channel index pid_c + C_half)
    val = tl.load(x_ptr + pid_n * x_stride_n + (pid_c + C_half) * x_stride_c + pid_t * x_stride_t)
    tl.store(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t, val)


@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True: add, False: subtract
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x1_val = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    res = x1_val + h_val if ADD else x1_val - h_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


@triton.jit
def cat_halves_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, T,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, 2*C_half, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    if pid_c < C_half:
        val = tl.load(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t)
    else:
        val = tl.load(x1_ptr + pid_n * x1_stride_n + (pid_c - C_half) * x1_stride_c + pid_t * x1_stride_t)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, val)


@triton.jit
def mask_mul_kernel(
    h_ptr, mask_ptr, out_ptr,
    N, C, T,
    h_stride_n, h_stride_c, h_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    mask_val = tl.load(mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + pid_t * mask_stride_t)  # mask has shape [N, 1, T]
    out_val = h_val * mask_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool,
        # 4 transforms, each with 3 conv weights and biases
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
        # Ensure contiguous tensors
        x = x.contiguous()
        x_mask = x_mask.contiguous()

        N, C, T = x.shape
        assert C == 192, "Expected C=192"
        half_channels = C // 2  # 96

        # We will perform forward in loops for 4 transforms.
        # Note: In original code, x_mask is [N, 1, T] and broadcast to channels. We apply it to h.

        # Prepare output tensors for forward/reverse
        # We'll keep x1 buffer updated in each transform loop.
        x1 = x[:, half_channels:, :].contiguous()

        # Loop over 4 transforms
        # We need x0 each time; we split x dynamically in each loop.
        for t_idx in range(4):
            # Determine which set of weights to use based on t_idx (positional args imply 4 transforms)
            # Python dispatch here is fine: args are positional, we just access them.
            conv0_w = locals()[f"transform_{t_idx}_conv0_weight"]
            conv0_b = locals()[f"transform_{t_idx}_conv0_bias"]
            conv1_w = locals()[f"transform_{t_idx}_conv1_weight"]
            conv1_b = locals()[f"transform_{t_idx}_conv1_bias"]
            conv2_w = locals()[f"transform_{t_idx}_conv2_weight"]
            conv2_b = locals()[f"transform_{t_idx}_conv2_bias"]

            # x0 is the first half of channels
            x0 = x[:, :half_channels, :].contiguous()

            # conv0: y0 = conv1d(x0, conv0_w, conv0_b)
            C_in0 = x0.shape[1]
            C_out0 = conv0_w.shape[0]
            K0 = conv0_w.shape[2]
            T_out0 = T - K0 + 1  # padding=0 behavior
            y0 = torch.empty((N, C_out0, T_out0), device=x.device, dtype=x.dtype)

            grid0 = (N, triton.cdiv(C_out0, 64), T_out0)
            conv1d_forward_kernel[grid0](
                x0, conv0_w, conv0_b, y0,
                N, C_in0, T, C_out0, T_out0, K0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_C=64,
            )

            # ReLU
            y0_relu = torch.empty_like(y0)
            grid_relu0 = (N, C_out0, T_out0)
            conv1d_relu_kernel[grid_relu0](
                y0, y0_relu,
                N, C_out0, T_out0,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            )

            # conv1: y1 = conv1d(y0_relu, conv1_w, conv1_b)
            C_in1 = y0_relu.shape[1]
            C_out1 = conv1_w.shape[0]
            K1 = conv1_w.shape[2]
            T_out1 = T_out0 - K1 + 1
            y1 = torch.empty((N, C_out1, T_out1), device=x.device, dtype=x.dtype)

            grid1 = (N, triton.cdiv(C_out1, 64), T_out1)
            conv1d_forward_kernel[grid1](
                y0_relu, conv1_w, conv1_b, y1,
                N, C_in1, T_out0, C_out1, T_out1, K1,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_C=64,
            )

            # ReLU
            y1_relu = torch.empty_like(y1)
            grid_relu1 = (N, C_out1, T_out1)
            conv1d_relu_kernel[grid_relu1](
                y1, y1_relu,
                N, C_out1, T_out1,
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
            )

            # conv2: h = conv1d(y1_relu, conv2_w, conv2_b)
            C_in2 = y1_relu.shape[1]
            C_out2 = conv2_w.shape[0]
            K2 = conv2_w.shape[2]
            T_out2 = T_out1 - K2 + 1
            h = torch.empty((N, C_out2, T_out2), device=x.device, dtype=x.dtype)

            grid2 = (N, triton.cdiv(C_out2, 64), T_out2)
            conv1d_forward_kernel[grid2](
                y1_relu, conv2_w, conv2_b, h,
                N, C_in2, T_out1, C_out2, T_out2, K2,
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_C=64,
            )

            # Apply mask (broadcast over channels)
            h_masked = torch.empty_like(h)
            grid_mask = (N, C_out2, T_out2)
            mask_mul_kernel[grid_mask](
                h, x_mask, h_masked,
                N, C_out2, T_out2,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            )

            # Update x1: forward adds, reverse subtracts
            x1_new = torch.empty_like(x1)
            grid_add = (N, half_channels, T_out2)
            add_halves_kernel[grid_add](
                x1, h_masked, x1_new,
                N, half_channels, T_out2,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                x1_new.stride(0), x1_new.stride(1), x1_new.stride(2),
                ADD=True if not reverse else False,
            )

            # Replace x1 buffer for next iteration
            x1 = x1_new

            # After each transform, x0 is still the first half of original x, but we update x1. For next iteration,
            # we keep splitting x as (x0 unchanged, x1 updated). However, since x1 is updated, we must ensure the
            # next iteration uses the latest x1 in case coupling is cumulative. In this task, the coupling is per
            # transform independent (each transform updates its own x1 based on x0), so we don't need to carry
            # previous x1. We will re-split x as original each iteration. If a true residual coupling were required,
            # we would need to keep track of updated x1 across transforms, but the provided code does not do that.

        # Final concatenation: x_out = [x0, x1]
        x_out = torch.empty((N, C, T_out2), device=x.device, dtype=x.dtype)
        grid_cat = (N, 2 * half_channels, T_out2)
        cat_halves_kernel[grid_cat](
            x[:, :half_channels, :], x1, x_out,
            N, half_channels, T_out2,
            x[:, :half_channels, :].stride(0), x[:, :half_channels, :].stride(1), x[:, :half_channels, :].stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
        )

        return x_out


def run(*args):
    return ModelNew()(*args)
