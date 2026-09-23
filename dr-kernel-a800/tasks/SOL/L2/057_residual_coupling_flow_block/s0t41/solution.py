import math
import torch
import torch.nn as nn
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
    # Grid dims: (N, C_out, T_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_tout = tl.program_id(2)

    co_offsets = pid_co + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = pid_tout - k  # output time index minus kernel tap
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x[n, ci, t_in] for all co in the block
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            x_ptrs = x_ptr + x_offsets + co_offsets * x_stride_c
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            # Load w[co, ci, k] for all co in the block
            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    # Add bias b[co] for this block
    b_ptrs = b_ptr + co_offsets
    b_vals = tl.load(b_ptrs, mask=co_mask, other=0.0)
    acc += b_vals

    # Store result
    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_tout * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


@triton.jit
def conv1d_relu_kernel(
    inp_ptr, out_ptr,
    N, C_out, T_out,
    inp_stride_n, inp_stride_c, inp_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid dims: (N, C_out, T_out)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    inp_offset = pid_n * inp_stride_n + pid_c * inp_stride_c + pid_t * inp_stride_t
    val = tl.load(inp_ptr + inp_offset)
    val = tl.maximum(val, 0.0)  # ReLU
    out_offset = pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offset, val)


@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True for forward (add), False for reverse (subtract)
):
    # Grid dims: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x1_val = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    if ADD:
        res = x1_val + h_val
    else:
        res = x1_val - h_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


@triton.jit
def mask_mul_kernel(
    h_ptr, mask_ptr, out_ptr,
    N, C, T,
    h_stride_n, h_stride_c, h_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid dims: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    # mask has shape [N, 1, T], so we load mask[n, 0, t]
    m_val = tl.load(mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + pid_t * mask_stride_t)
    res = h_val * m_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                # 4 transforms
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
        """
        Triton-optimized forward. Implements Conv1d (forward + ReLU) and coupling (add/subtract) in Triton.
        Heavy compute (conv + relu) is performed by Triton kernels; mask multiplication is also Triton.
        Minimal slicing/concatenation is done with torch to build the final output.
        """
        N, C, T = x.shape
        half_channels = C // 2  # 96

        # Split into halves
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

        # Helper function to compute one transform using Triton convs + ReLU, return h
        def apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
            # conv0 forward
            C_in0 = x0.shape[1]
            C_out0 = conv0_w.shape[0]
            K0 = conv0_w.shape[2]
            T_in0 = x0.shape[2]
            T_out0 = T_in0 - K0 + 1

            y0 = torch.empty((N, C_out0, T_out0), device=x.device, dtype=torch.float32)
            grid0 = (N, C_out0, T_out0)
            conv1d_forward_kernel[grid0](
                x0, conv0_w, conv0_b, y0,
                N, C_in0, T_in0, C_out0, T_out0, K0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_C=64,
            )

            # ReLU conv0 output
            y0_relu = torch.empty_like(y0)
            conv1d_relu_kernel[grid0](
                y0, y0_relu,
                N, C_out0, T_out0,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
            )

            # conv1 forward
            C_in1 = C_out0
            C_out1 = conv1_w.shape[0]
            K1 = conv1_w.shape[2]
            T_in1 = y0_relu.shape[2]
            T_out1 = T_in1 - K1 + 1

            y1 = torch.empty((N, C_out1, T_out1), device=x.device, dtype=torch.float32)
            grid1 = (N, C_out1, T_out1)
            conv1d_forward_kernel[grid1](
                y0_relu, conv1_w, conv1_b, y1,
                N, C_in1, T_in1, C_out1, T_out1, K1,
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_C=64,
            )

            # ReLU conv


def run(*args):
    return ModelNew()(*args)
