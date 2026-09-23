import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all computations must be inside these
if TRITON_AVAILABLE:
    @triton.jit
    def conv1d_forward_kernel(
        x_ptr,         # *float32, [N, C_IN, T_IN]
        w_ptr,         # *float32, [C_OUT, C_IN, K]
        b_ptr,         # *float32, [C_OUT]
        y_ptr,         # *float32, [N, C_OUT, T_OUT]
        N, T_IN, T_OUT, C_IN: tl.constexpr, C_OUT: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        t_block_start: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # program ids
        pid_n = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_tb = tl.program_id(2)

        # time offsets this program computes
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_OUT

        # accumulator for this (n, co, t_block)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # loop over input channels and kernel taps
        for ci in range(0, C_IN):
            for k in range(0, K):
                t_in = t_offsets + k - PAD
                valid = (t_in >= 0) & (t_in < T_IN) & t_mask
                # load x[n, ci, t_in] with mask
                x_ptrs = x_ptr + pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
                # load weight w[co, ci, k]
                w_ptrs = w_ptr + pid_co * w_stride_co + ci * w_stride_ci + k * w_stride_k
                w_val = tl.load(w_ptrs)
                acc += x_vals * w_val

        # add bias for this output channel
        b_val = tl.load(b_ptr + pid_co)
        acc += b_val

        # store y[n, co, t_offsets]
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_co * y_stride_c + t_offsets * y_stride_t
        tl.store(y_ptrs, acc, mask=t_mask)

    @triton.jit
    def relu_forward_kernel(
        inp_ptr,        # *float32, [N, C, T] (input tensor)
        out_ptr,        # *float32, [N, C, T] (output tensor, can alias inp_ptr)
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C
        t_block_start = pid_tb * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        mask = t_offsets < T

        in_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t_offsets * in_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        x = tl.load(in_ptrs, mask=mask, other=0.0)
        x = tl.maximum(x, 0.0)
        tl.store(out_ptrs, x, mask=mask)

    @triton.jit
    def concat_half_channels_kernel_fixed(
        x0_ptr,         # *float32, [N, C_HALF, T]
        x1_ptr,         # *float32, [N, C_HALF, T]  (second half channels, length C_HALF)
        out_ptr,        # *float32, [N, 2*C_HALF, T]
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        # grid: (N, 2*C_HALF, ceil_div(T, BLOCK_T))
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_tb = tl.program_id(2)

        t_block_start = pid_tb * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        # first half channels
        if pid_c < C_HALF:
            in_ptrs = x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + t_offsets * x0_stride_t
            out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t
            x = tl.load(in_ptrs, mask=t_mask, other=0.0)
            tl.store(out_ptrs, x, mask=t_mask)
        # second half channels (shift by C_HALF)
        else:
            ci = pid_c - C_HALF
            in_ptrs = x1_ptr + pid_n * x1_stride_n + ci * x1_stride_c + t_offsets * x1_stride_t
            out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t
            x = tl.load(in_ptrs, mask=t_mask, other=0.0)
            tl.store(out_ptrs, x, mask=t_mask)

    @triton.jit
    def add_affine_kernel(
        src_ptr,        # *float32, [N, C, T] (x1 before coupling)
        add_ptr,        # *float32, [N, C, T] (h from transform)
        out_ptr,        # *float32, [N, C, T] (x1 after coupling)
        N, C, T,
        src_stride_n, src_stride_c, src_stride_t,
        add_stride_n, add_stride_c, add_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C
        t_block_start = pid_tb * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        mask = t_offsets < T

        src_ptrs = src_ptr + n * src_stride_n + c * src_stride_c + t_offsets * src_stride_t
        add_ptrs = add_ptr + n * add_stride_n + c * add_stride_c + t_offsets * add_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        a = tl.load(src_ptrs, mask=mask, other=0.0)
        b = tl.load(add_ptrs, mask=mask, other=0.0)
        c_out = a + b
        tl.store(out_ptrs, c_out, mask=mask)

    @triton.jit
    def sub_affine_kernel(
        src_ptr,        # *float32, [N, C, T] (x1 before coupling)
        sub_ptr,        # *float32, [N, C, T] (h from transform)
        out_ptr,        # *float32, [N, C, T] (x1 after coupling)
        N, C, T,
        src_stride_n, src_stride_c, src_stride_t,
        sub_stride_n, sub_stride_c, sub_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C
        t_block_start = pid_tb * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        mask = t_offsets < T

        src_ptrs = src_ptr + n * src_stride_n + c * src_stride_c + t_offsets * src_stride_t
        sub_ptrs = sub_ptr + n * sub_stride_n + c * sub_stride_c + t_offsets * sub_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        a = tl.load(src_ptrs, mask=mask, other=0.0)
        b = tl.load(sub_ptrs, mask=mask, other=0.0)
        c_out = a - b
        tl.store(out_ptrs, c_out, mask=mask)

    @triton.jit
    def mask_mul_kernel(
        inp_ptr,        # *float32, [N, C, T]
        mask_ptr,       # *float32, [N, 1, T] (per sample, per time)
        out_ptr,        # *float32, [N, C, T]
        N, C, T,
        inp_stride_n, inp_stride_c, inp_stride_t,
        mask_stride_n, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C
        t_block_start = pid_tb * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        mask = t_offsets < T

        inp_ptrs = inp_ptr + n * inp_stride_n + c * inp_stride_c + t_offsets * inp_stride_t
        mask_ptrs = mask_ptr + n * mask_stride_n + t_offsets * mask_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        x = tl.load(inp_ptrs, mask=mask, other=0.0)
        m = tl.load(mask_ptrs, mask=mask, other=1.0)
        y = x * m
        tl.store(out_ptrs, y, mask=mask)


# Triton-only ModelNew: forward uses only Triton kernels
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we will receive inputs in forward

    def forward(self, x: torch.Tensor,
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
                transform_3_conv2_bias: torch.Tensor):
        # All computation must be in Triton. Ensure CUDA and Triton availability.
        assert TRITON_AVAILABLE and x.is_cuda, "ModelNew.forward requires Triton and CUDA tensors."

        # Define constants and splits
        N, C, T = x.shape
        assert C == 192, "Expected channels=192"
        half_channels = 96
        convs = [
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

        # Work in float32 for simplicity
        x = x.contiguous().to(torch.float32)

        # Forward vs reverse loop
        if not reverse:
            for i, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(convs):
                # x0: first half channels
                x0 = x[:, :half_channels, :]
                # h = apply_transform(x0) = conv1d -> ReLU -> conv1d -> ReLU -> conv1d
                # We implement the full transform in Triton.

                # conv0: [C_IN=96, K=5], out_channels = conv0_w.shape[0] = 192
                C_IN0 = conv0_w.shape[1]
                K0 = conv0_w.shape[2]
                C_OUT0 = conv0_w.shape[0]
                PAD0 = K0 // 2
                T_IN0 = T
                T_OUT0 = T_IN0  # preserved with padding

                # Allocate y0
                y0 = torch.empty((N, C_OUT0, T_OUT0), device=x.device, dtype=torch.float32)
                # Launch conv1d kernel
                grid0 = (N, C_OUT0, (T_OUT0 + 127) // 128)
                conv1d_forward_kernel[grid0](
                    x0, conv0_w, conv0_b, y0,
                    N, T_IN0, T_OUT0, C_IN0, C_OUT0, K0, PAD0,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    0, 128
                )

                # ReLU
                y0_relu = torch.empty_like(y0)
                grid_relu = (N * C_OUT0, (T_OUT0 + 127) // 128)
                relu_forward_kernel[grid_relu](
                    y0, y0_relu, N, C_OUT0, T_OUT0,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                    *grid_relu
                )
                # conv1: [C_IN=C_OUT0=192, K=5], out_channels = 192
                C_IN1 = conv1_w.shape[1]
                K1 = conv1_w.shape[2]
                C_OUT1 = conv1_w.shape[0]
                PAD1 = K1 // 2
                y1 = torch.empty((N, C_OUT1, T_OUT0), device=x.device, dtype=torch.float32)
                grid1 = (N, C_OUT1, (T_OUT0 + 127) // 128)
                conv1d_forward_kernel[grid1](
                    y0_relu, conv1_w, conv1_b, y1,
                    N, y0_relu.shape[1], T_OUT0, C_IN1, C_OUT1, K1, PAD1,
                    y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    0, 128
                )

                # ReLU
                y1_relu = torch.empty_like(y1)
                grid2 = (N * C_OUT1, (T_OUT0 + 127) // 128)
                relu_forward_kernel[grid2](
                    y1, y1_relu, N, C_OUT1, T_OUT0,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                    *grid2
                )

                # conv2: [C_IN=C_OUT1=192, K=5], out_channels = 96
                C_IN2 = conv2_w.shape[1]
                K2 = conv2_w.shape[2]
                C_OUT2 = conv2_w.shape[0]
                PAD2 = K2 // 2
                y2 = torch.empty((N, C_OUT2, T_OUT0), device=x.device, dtype=torch.float32)
                grid3 = (N, C_OUT2, (T_OUT0 + 127) // 128)
                conv1d_forward_kernel[grid3](
                    y1_relu, conv2_w, conv2_b, y2,
                    N, y1_relu.shape[1], T_OUT0, C_IN2, C_OUT2, K2, PAD2,
                    y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    y2.stride(0), y2.stride(1), y2.stride(2),
                    0, 128
                )

                # h = y2, shape [N, 96, T]
                # Apply mask (generic, though x_mask is ones here)
                h = y2
                # mask multiply (x_mask is [N, 1, T])
                h_masked = torch.empty_like(h)
                grid_mask = (N * h.shape[1], (T_OUT0 + 127) // 128)
                mask_mul_kernel[grid_mask](
                    h, x_mask, h_masked, N, h.shape[1], T_OUT0,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    *grid_mask
                )
                h = h_masked

                # Update second half channels: x1 += h
                x1 = x[:, half_channels:, :]
                out_x1 = torch.empty_like(x1)
                grid_aff = (N * x1.shape[1], (T_OUT0 + 127) // 128)
                add_affine_kernel[grid_aff](
                    x1, h, out_x1, N, x1.shape[1], T_OUT0,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                    *grid_aff
                )
                # Assign updated x1 back to x
                x[:, half_channels:] = out_x1

                # Apply mask to entire x (x_mask is [N, 1, T])
                x_masked = torch.empty_like(x)
                grid_mask_all = (N * C, (T_OUT0 + 127) // 128)
                mask_mul_kernel[grid_mask_all](
                    x, x_mask, x_masked, N, C, T_OUT0,
                    x.stride(0), x.stride(1), x.stride(2),
                    x_mask.stride(0), x_mask.stride(2),
                    x_masked.stride(0), x_masked.stride(1), x_masked.stride(2),
                    *grid_mask_all
                )
                x = x_masked

        else:
            # Reverse pass: x1 = x1 - h in reversed order of transforms
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(convs):
                x0 = x[:, :half_channels, :]
                x1 = x[:, half_channels:, :]

                # Forward transform of x0: conv0 -> ReLU -> conv1 -> ReLU -> conv2
                C_IN0 = conv0_w.shape[1]
                K0 = conv0_w.shape[2]
                C_OUT0 = conv0_w.shape[0]
                PAD0 = K0 // 2

                y0 = torch.empty((N, C_OUT0, T), device=x.device, dtype=torch.float32)
                grid0 = (N, C_OUT0, (T + 127) // 128)
                conv1d_forward_kernel[grid0](
                    x0, conv0_w, conv0_b, y0,
                    N, x0.shape[1], T, C_IN0, C_OUT0, K0, PAD0,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    0, 128
                )

                y0_relu = torch.empty_like(y0)
                grid_relu0 = (N * C_OUT0, (T + 127) // 128)
                relu_forward_kernel[grid_relu0](
                    y0, y0_relu, N, C_OUT0, T,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                    *grid_relu0
                )

                C_IN1 = conv1_w.shape[1]
                K1 = conv1_w.shape[2]
                C_OUT1 = conv1_w.shape[0]
                PAD1 = K1 // 2

                y1 = torch.empty((N, C_OUT1, T), device=x.device, dtype=torch.float32)
                grid1 = (N, C_OUT1, (T + 127) // 128)
                conv1d_forward_kernel[grid1](
                    y0_relu, conv1_w, conv1_b, y1,
                    N, y0_relu.shape[1], T, C_IN1, C_OUT1, K1, PAD1,
                    y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    0, 128
                )

                y1_relu = torch.empty_like(y1)
                grid2 = (N * C_OUT1, (T + 127) // 128)
                relu_forward_kernel[grid2](
                    y1, y1_relu, N, C_OUT1, T,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                    *grid2
                )

                C_IN2 = conv2_w.shape[1]
                K2 = conv2_w.shape[2]
                C_OUT2 = conv2_w.shape[0]
                PAD2 = K2 // 2

                y2 = torch.empty((N, C_OUT2, T), device=x.device, dtype=torch.float32)
                grid3 = (N, C_OUT2, (T + 127) // 128)
                conv1d_forward_kernel[grid3](
                    y1_relu, conv2_w, conv2_b, y2,
                    N, y1_relu.shape[1], T, C_IN2, C_OUT2, K2, PAD2,
                    y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    y2.stride(0), y2.stride(1), y2.stride(2),
                    0, 128
                )

                h = y2
                h_masked = torch.empty_like(h)
                grid_mask_h = (N * h.shape[1], (T + 127) // 128)
                mask_mul_kernel[grid_mask_h](
                    h, x_mask, h_masked, N, h.shape[1], T,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    *grid_mask_h
                )
                h = h_masked

                # Affine coupling: x1 = x1 - h
                out_x1 = torch.empty_like(x1)
                grid_aff = (N * x1.shape[1], (T + 127) // 128)
                sub_affine_kernel[grid_aff](
                    x1, h, out_x1, N, x1.shape[1], T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                    *grid_aff
                )
                x[:, half_channels:] = out_x1

                # Apply mask to entire x
                x_masked = torch.empty_like(x)
                grid_mask_all = (N * C, (T + 127) // 128)
                mask_mul_kernel[grid_mask_all](
                    x, x_mask, x_masked, N, C, T,
                    x.stride(0), x.stride(1), x.stride(2),
                    x_mask.stride(0), x_mask.stride(2),
                    x_masked.stride(0), x_masked.stride(1), x_masked.stride(2),
                    *grid_mask_all
                )
                x = x_masked

        return x


def run(*args):
    return ModelNew()(*args)
