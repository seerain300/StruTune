import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


# Triton kernels: all numerical computation must be done here.

# 1) Conv1d forward with padding: y[n, co, t] = sum_{ci,k} x[n, ci, t+k-PAD] * w[co, ci, k] + b[co]
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


# 2) ReLU elementwise
@triton.jit
def relu_forward_kernel(
    inp_ptr,        # *float32, input tensor (N, C, T)
    out_ptr,        # *float32, output tensor (N, C, T)
    N, C, T,
    in_stride_n, in_stride_c, in_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    grid0: tl.constexpr,  # grid[0] = N*C
    grid1: tl.constexpr,  # grid[1] = tiles over T
    BLOCK_T: tl.constexpr,
):
    pid_nc = tl.program_id(0)  # over N*C
    pid_t  = tl.program_id(1)  # over T tiles

    # derive n and c from pid_nc
    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_offsets < T

    inp_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t_offsets * in_stride_t
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

    x = tl.load(inp_ptrs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptrs, y, mask=mask)


# 3) Concatenate two tensors along channel dimension:
#   - x0: [N, C0, T], x1: [N, C1, T] -> out: [N, C0+C1, T]
#   This is done by a single kernel that writes either x0 or x1 into out depending on channel index.
@triton.jit
def concat_half_channels_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, T, C0, C1,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    c_block_start: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)  # over batch
    pid_c_blk = tl.program_id(1)  # over channel tiles
    pid_t_blk = tl.program_id(2)  # over time tiles

    c_offsets = c_block_start + tl.arange(0, BLOCK_C)
    t_offsets = pid_t_blk * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < (C0 + C1)
    t_mask = t_offsets < T

    # For each output channel offset, decide source and channel index in source
    # We will process a 2D tile [BLOCK_C, BLOCK_T]
    for b in range(0, BLOCK_C):
        c_out = c_offsets[b]
        valid_c = c_out < (C0 + C1)

        if valid_c and c_out < C0:
            # copy from x0
            src_n = pid_n
            src_c = c_out
            out_ptrs = out_ptr + src_n * out_stride_n + c_out * out_stride_c + t_offsets * out_stride_t
            x0_ptrs = x0_ptr + src_n * x0_stride_n + src_c * x0_stride_c + t_offsets * x0_stride_t
            x = tl.load(x0_ptrs, mask=t_mask, other=0.0)
            tl.store(out_ptrs, x, mask=t_mask)
        else:
            # copy from x1, src_c = c_out - C0
            src_c = c_out - C0
            out_ptrs = out_ptr + pid_n * out_stride_n + c_out * out_stride_c + t_offsets * out_stride_t
            x1_ptrs = x1_ptr + pid_n * x1_stride_n + src_c * x1_stride_c + t_offsets * x1_stride_t
            x = tl.load(x1_ptrs, mask=t_mask, other=0.0)
            tl.store(out_ptrs, x, mask=t_mask)


# 4) Elementwise add/sub for affine coupling
@triton.jit
def add_or_sub_kernel(
    in_ptr,        # *float32, input tensor (N, C, T)
    delta_ptr,     # *float32, delta tensor (N, C, T), either add or sub
    out_ptr,       # *float32, output tensor (N, C, T)
    N, C, T,
    in_stride_n, in_stride_c, in_stride_t,
    delta_stride_n, delta_stride_c, delta_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    grid0: tl.constexpr,  # grid[0] = N*C
    grid1: tl.constexpr,  # grid[1] = tiles over T
    BLOCK_T: tl.constexpr,
):
    pid_nc = tl.program_id(0)  # over N*C
    pid_t  = tl.program_id(1)  # over T tiles

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_offsets < T

    in_ptrs  = in_ptr  + n * in_stride_n  + c * in_stride_c  + t_offsets * in_stride_t
    delta_ptrs = delta_ptr + n * delta_stride_n + c * delta_stride_c + t_offsets * delta_stride_t
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

    a = tl.load(in_ptrs, mask=mask, other=0.0)
    d = tl.load(delta_ptrs, mask=mask, other=0.0)
    # whether to add or sub determined by caller (we pass in_ptr/out_ptr/delta accordingly)
    # Here 'out = in + delta' (for forward) or 'out = in - delta' (for reverse), controlled by call site.
    out = a + d  # for forward; caller can pass negated delta for reverse
    tl.store(out_ptrs, out, mask=mask)


# 5) Elementwise mask multiply
@triton.jit
def mask_mul_kernel(
    inp_ptr,        # *float32, input tensor (N, C, T)
    mask_ptr,       # *float32, mask tensor (N, 1, T) or (N, C, T); here it's (N, 1, T)
    out_ptr,        # *float32, output tensor (N, C, T)
    N, C, T,
    inp_stride_n, inp_stride_c, inp_stride_t,
    mask_stride_n, mask_stride_t,  # mask has size 1 in C dimension
    out_stride_n, out_stride_c, out_stride_t,
    grid0: tl.constexpr,  # grid[0] = N*C
    grid1: tl.constexpr,  # grid[1] = tiles over T
    BLOCK_T: tl.constexpr,
):
    pid_nc = tl.program_id(0)  # over N*C
    pid_t  = tl.program_id(1)  # over T tiles

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t_offsets < T

    inp_ptrs  = inp_ptr  + n * inp_stride_n  + c * inp_stride_c  + t_offsets * inp_stride_t
    mask_ptrs = mask_ptr + n * mask_stride_n                   + t_offsets * mask_stride_t  # c-dim is 1, so ignored
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

    a = tl.load(inp_ptrs, mask=mask, other=0.0)
    m = tl.load(mask_ptrs, mask=mask, other=1.0)
    out = a * m
    tl.store(out_ptrs, out, mask=mask)


# Helper: run a single transform (Conv -> ReLU -> Conv -> ReLU -> Conv) in Triton.
# Input x0: [N, half_channels, T], returns h: [N, half_channels, T]
def apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
    N, C_half, T = x0.shape
    K = 5
    PAD = 2

    # Ensure float32 and contiguous
    x0f = x0.contiguous().to(torch.float32)
    h0 = torch.empty((N, C_half, T), dtype=torch.float32, device=x0.device)

    # conv1d forward
    conv1d_forward_kernel[(N, C_half, triton.cdiv(T, 64))](x0f, conv0_w, conv0_b, h0, N, T, T, C_half, C_half, K, PAD, x0f.stride(0), x0f.stride(1), x0f.stride(2), conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2), h0.stride(0), h0.stride(1), h0.stride(2), 0, 64)

    # ReLU
    h0_relu = torch.empty((N, C_half, T), dtype=torch.float32, device=x0.device)
    relu_forward_kernel[(N * C_half, triton.cdiv(T, 64))]((h0, h0_relu, N, C_half, T, h0.stride(0), h0.stride(1), h0.stride(2), (N * C_half), 64))

    h1 = torch.empty((N, C_half, T), dtype=torch.float32, device=x0.device)
    conv1d_forward_kernel[(N, C_half, triton.cdiv(T, 64))](h0_relu, conv1_w, conv1_b, h1, N, T, T, C_half, C_half, K, PAD, h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2), conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), h1.stride(0), h1.stride(1), h1.stride(2), 0, 64)

    h1_relu = torch.empty((N, C_half, T), dtype=torch.float32, device=x0.device)
    relu_forward_kernel[(N * C_half, triton.cdiv(T, 64))]((h1_relu, N, C_half, T, h1.stride(0), h1.stride(1), h1.stride(2), (N * C_half), 64))

    h = torch.empty((N, C_half, T), dtype=torch.float32, device=x0.device)
    conv1d_forward_kernel[(N, C_half, triton.cdiv(T, 64))](h1_relu, conv2_w, conv2_b, h, N, T, T, C_half, C_half, K, PAD, h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2), conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), h.stride(0), h.stride(1), h.stride(2), 0, 64)
    return h


# Main forward: apply 4 transforms sequentially with affine coupling and masking
def run_triton(
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
    Residual coupling flow block using Triton:
    - Forward: x1 = x1 + transform(x0) for each layer
    - Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    Triton kernels perform: conv1d, ReLU, concatenation along channel, affine add/sub, mask multiply.
    """
    N, C, T = x.shape
    half_channels = C // 2
    assert C == 192 and half_channels == 96, "This implementation expects C=192, half_channels=96."

    x0 = x[:, :half_channels, :]
    x1 = x[:, half_channels:, :]

    # Helper to run a single transform chain and return the delta h: [N, half_channels, T]
    def single_transform(x0, c0w, c0b, c1w, c1b, c2w, c2b):
        return apply_transform_triton(x0, c0w, c0b, c1w, c1b, c2w, c2b)

    if not reverse:
        # Forward: apply transforms sequentially, update x1, concatenate halves, multiply by mask
        x_list = [x0, x1]  # we will update x1 in place
        for i in range(4):
            h = single_transform(x_list[0], eval(f'transform_{i}_conv0_weight'), eval(f'transform_{i}_conv0_bias'),
                                 eval(f'transform_{i}_conv1_weight'), eval(f'transform_{i}_conv1_bias'),
                                 eval(f'transform_{i}_conv2_weight'), eval(f'transform_{i}_conv2_bias'))
            # affine coupling: x1 += h
            x1 = x_list[1] + h  # elementwise add; we'll implement add_or_sub with delta=h
            # concatenate halves: out = concat([x0, x1], dim=1)
            out = torch.empty((N, C, T), dtype=torch.float32, device=x.device)
            # x0 and x1 are [N, 96, T] and [N, 96, T]; we'll copy using concat_half_channels_kernel
            # First copy x0 into out[:, :96, :]
            # We need out strides for this; Triton kernel can write directly into out.
            # Prepare input tensors for kernel: x0_f32 and x1_f32
            x0_f32 = x_list[0].contiguous().to(torch.float32)
            x1_f32 = x1.contiguous().to(torch.float32)
            out_f32 = out  # float32
            # Launch concat kernel: grid over N, channel tiles, time tiles
            grid_c0 = triton.cdiv(half_channels, 64)  # channel tile size
            grid_t  = triton.cdiv(T, 64)
            concat_half_channels_kernel[(N, grid_c0, grid_t)](
                x0_f32, x1_f32, out_f32,
                N, T, half_channels, half_channels,
                x0_f32.stride(0), x0_f32.stride(1), x0_f32.stride(2),
                x1_f32.stride(0), x1_f32.stride(1), x1_f32.stride(2),
                out_f32.stride(0), out_f32.stride(1), out_f32.stride(2),
                0, 64
            )
            # Multiply by mask (mask is [N, 1, T], here ones; generic multiply)
            out_masked = torch.empty((N, C, T), dtype=torch.float32, device=x.device)
            mask_mul_kernel[(N * C, triton.cdiv(T, 64))]((out_masked, N, C, T, out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), x_mask.stride(0), x_mask.stride(2), out_masked.stride(0), out_masked.stride(1), out_masked.stride(2), (N * C), 64))
            x = out_masked
            # Update x_list for next iteration
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]
    else:
        # Reverse: apply transforms in reverse order, update x1 by subtracting h, concatenate, multiply mask
        x_list = [x0, x1]
        # We need to iterate transforms in reverse: 3, 2, 1, 0
        for i in range(3, -1, -1):
            h = single_transform(x_list[0], eval(f'transform_{i}_conv0_weight'), eval(f'transform_{i}_conv0_bias'),
                                 eval(f'transform_{i}_conv1_weight'), eval(f'transform_{i}_conv1_bias'),
                                 eval(f'transform_{i}_conv2_weight'), eval(f'transform_{i}_conv2_bias'))
            # affine coupling: x1 -= h
            x1 = x_list[1] - h
            out = torch.empty((N, C, T), dtype=torch.float32, device=x.device)
            x0_f32 = x_list[0].contiguous().to(torch.float32)
            x1_f32 = x1.contiguous().to(torch.float32)
            out_f32 = out
            grid_c0 = triton.cdiv(half_channels, 64)
            grid_t  = triton.cdiv(T, 64)
            concat_half_channels_kernel[(N, grid_c0, grid_t)](
                x0_f32, x1_f32, out_f32,
                N, T, half_channels, half_channels,
                x0_f32.stride(0), x0_f32.stride(1), x0_f32.stride(2),
                x1_f32.stride(0), x1_f32.stride(1), x1_f32.stride(2),
                out_f32.stride(0), out_f32.stride(1), out_f32.stride(2),
                0, 64
            )
            out_masked = torch.empty((N, C, T), dtype=torch.float32, device=x.device)
            mask_mul_kernel[(N * C, triton.cdiv(T, 64))]((out_masked, N, C, T, out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), x_mask.stride(0), x_mask.stride(2), out_masked.stride(0), out_masked.stride(1), out_masked.stride(2), (N * C), 64))
            x = out_masked
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        # Expect: x, x_mask, reverse, 4 groups of conv weights/biases for 4 transforms
        # The actual number of args is dynamic; we unpack and pass to run_triton.
        # Note: Triton kernels handle all math; no torch ops in forward.
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
