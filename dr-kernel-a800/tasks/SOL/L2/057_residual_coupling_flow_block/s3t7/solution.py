import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_bias_stride1_kernel(
    x_ptr,  # *const T
    w_ptr,  # *const T
    b_ptr,  # *const float32
    y_ptr,  # *T
    N: tl.constexpr,  # int
    Cin: tl.constexpr,  # int
    Cout: tl.constexpr,  # int
    L_in: tl.constexpr,  # int
    L_out: tl.constexpr,  # int
    K: tl.constexpr,  # int (kernel_size), here 5
    x_stride_n, x_stride_c, x_stride_l,  # int
    w_stride_co, w_stride_ci, w_stride_k,  # int
    y_stride_n, y_stride_c, y_stride_l,  # int
):
    """
    Compute y[n, co, t_out] = conv1d(x[n, :, :], w[co, :, :], bias[b], stride=1, padding=0) for t_out in [0..L_out-1]
    x: [N, Cin, L_in], w: [Cout, Cin, K], y: [N, Cout, L_out]
    Assumes padding=0, so L_out = L_in - K + 1.
    """
    pid_nc = tl.program_id(0)  # over N * Cout
    pid_t = tl.program_id(1)   # over tiles of L_out

    n = pid_nc // Cout
    co = pid_nc % Cout

    # Output tile
    l_out_offsets = pid_t * 128 + tl.arange(0, 128)
    mask_out = l_out_offsets < L_out

    # Accumulator in fp32
    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over input channels and kernel taps
    # We assume Cin and K are small (e.g., 96 and 5), so nested loops are fine.
    for ci in range(Cin):
        for k in range(K):
            l_in_vec = l_out_offsets - k
            valid = (l_in_vec >= 0) & (l_in_vec < L_in) & mask_out
            x_ptrs = x_ptr + n * x_stride_n + ci * x_stride_c + l_in_vec * x_stride_l
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
            # Load scalar weight w[co, ci, k]
            w_ptrs = w_ptr + co * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_val = tl.load(w_ptrs).to(tl.float32)
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    # Store
    y_ptrs = y_ptr + n * y_stride_n + co * y_stride_c + l_out_offsets * y_stride_l
    tl.store(y_ptrs, acc, mask=mask_out)


@triton.jit
def relu_kernel(x_ptr, out_ptr, N, C, L, stride_n, stride_c, stride_l):
    pid0 = tl.program_id(0)  # over N*C
    pid1 = tl.program_id(1)  # over tiles of L
    n = pid0 // C
    c = pid0 % C
    l_offsets = pid1 * 128 + tl.arange(0, 128)
    mask_l = l_offsets < L
    x_ptrs = x_ptr + n * stride_n + c * stride_c + l_offsets * stride_l
    x_vec = tl.load(x_ptrs, mask=mask_l, other=0.0).to(tl.float32)
    out_vec = tl.maximum(x_vec, 0.0)
    out_ptrs = out_ptr + n * stride_n + c * stride_c + l_offsets * stride_l
    tl.store(out_ptrs, out_vec, mask=mask_l)


@triton.jit
def concatenate_channels_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, L,
    x0_stride_n, x0_stride_c, x0_stride_l,
    x1_stride_n, x1_stride_c, x1_stride_l,
    out_stride_n, out_stride_c, out_stride_l,
):
    """
    out[n, c, l] = x0[n, c, l] if c < C_half else x1[n, c - C_half, l]
    """
    pid0 = tl.program_id(0)  # over N * (2 * C_half)
    pid1 = tl.program_id(1)  # over tiles of L
    n = pid0 // (2 * C_half)
    c = pid0 % (2 * C_half)
    l_offsets = pid1 * 128 + tl.arange(0, 128)
    mask_l = l_offsets < L

    if c < C_half:
        in_ptrs = x0_ptr + n * x0_stride_n + c * x0_stride_c + l_offsets * x0_stride_l
    else:
        ci = c - C_half
        in_ptrs = x1_ptr + n * x1_stride_n + ci * x1_stride_c + l_offsets * x1_stride_l

    vals = tl.load(in_ptrs, mask=mask_l, other=0.0)
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + l_offsets * out_stride_l
    tl.store(out_ptrs, vals, mask=mask_l)


@triton.jit
def multiply_mask_kernel(
    x_ptr, mask_ptr, out_ptr,
    N, C, L,
    x_stride_n, x_stride_c, x_stride_l,
    mask_stride_n, mask_stride_l,  # mask has shape [N, 1, L]
):
    """
    out[n, c, l] = x[n, c, l] * mask[n, 0, l]
    """
    pid0 = tl.program_id(0)  # over N*C
    pid1 = tl.program_id(1)  # over tiles of L
    n = pid0 // C
    c = pid0 % C
    l_offsets = pid1 * 128 + tl.arange(0, 128)
    mask_l = l_offsets < L

    x_ptrs = x_ptr + n * x_stride_n + c * x_stride_c + l_offsets * x_stride_l
    mask_ptrs = mask_ptr + n * mask_stride_n + l_offsets * mask_stride_l  # channel stride ignored (mask has size 1)
    x_vec = tl.load(x_ptrs, mask=mask_l, other=0.0).to(tl.float32)
    mask_vec = tl.load(mask_ptrs, mask=mask_l, other=1.0).to(tl.float32)
    out_vec = x_vec * mask_vec
    out_ptrs = out_ptr + n * x_stride_n + c * x_stride_c + l_offsets * x_stride_l  # Note: reuse x strides for out write
    tl.store(out_ptrs, out_vec, mask=mask_l)


@triton.jit
def add_masked_kernel(
    x_ptr, h_ptr, out_ptr,
    N, C, L,
    x_stride_n, x_stride_c, x_stride_l,
    h_stride_n, h_stride_c, h_stride_l,
    op: tl.constexpr,  # 1 for add, 0 for sub
):
    """
    out[n, c, l] = x[n, c, l] + op * h[n, c, l]
    op = 1 if reverse==False else 0 -> subtract
    """
    pid0 = tl.program_id(0)  # over N*C
    pid1 = tl.program_id(1)  # over tiles of L
    n = pid0 // C
    c = pid0 % C
    l_offsets = pid1 * 128 + tl.arange(0, 128)
    mask_l = l_offsets < L

    x_ptrs = x_ptr + n * x_stride_n + c * x_stride_c + l_offsets * x_stride_l
    h_ptrs = h_ptr + n * h_stride_n + c * h_stride_c + l_offsets * h_stride_l
    x_vec = tl.load(x_ptrs, mask=mask_l, other=0.0).to(tl.float32)
    h_vec = tl.load(h_ptrs, mask=mask_l, other=0.0).to(tl.float32)
    if op:
        out_vec = x_vec + h_vec
    else:
        out_vec = x_vec - h_vec

    out_ptrs = out_ptr + n * x_stride_n + c * x_stride_c + l_offsets * x_stride_l
    tl.store(out_ptrs, out_vec, mask=mask_l)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Triton-optimized implementation of the original run():
        Accepts x [N, C, L], x_mask [N, 1, L], reverse (bool), then 48 tensors for 4 transforms:
        each transform has 3 weights and 3 biases in order: (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
        Returns transformed x.
        """
        # Ensure tensors are on CUDA
        device = args[0].device
        x = args[0].contiguous()
        x_mask = args[1].contiguous()
        reverse_flag = bool(args[2])

        # We will run 4 transforms, consistent with the original.
        half_channels = x.shape[1] // 2

        for _ in range(4):
            # Split into halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: [Cout=192, Cin=96, K=5], bias
            w0 = args[3].contiguous()
            b0 = args[4].contiguous()
            L_in = x0.shape[2]  # L
            # Output length for padding=0, K=5: L_out = L_in - 4
            L_out0 = L_in - 4
            h0 = torch.empty((x0.shape[0], w0.shape[0], L_out0), device=device, dtype=torch.float32)
            grid0 = (x0.shape[0] * w0.shape[0], triton.cdiv(L_out0, 128))
            conv1d_bias_stride1_kernel[grid0](
                x0, w0, b0, h0,
                x0.shape[0], x0.shape[1], w0.shape[0], x0.shape[2], L_out0, 5,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                num_warps=4
            )

            # ReLU
            h0 = triton_relu(h0)

            # conv1: [Cout=192, Cin=192, K=5], bias
            w1 = args[5].contiguous()
            b1 = args[6].contiguous()
            L_in1 = h0.shape[2]
            L_out1 = L_in1 - 4
            h1 = torch.empty((h0.shape[0], w1.shape[0], L_out1), device=device, dtype=torch.float32)
            grid1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, 128))
            conv1d_bias_stride1_kernel[grid1](
                h0, w1, b1, h1,
                h0.shape[0], w1.shape[1], w1.shape[0], h0.shape[2], L_out1, 5,
                h0.stride(0), h0.stride(1), h0.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                num_warps=4
            )

            # ReLU
            h1 = triton_relu(h1)

            # conv2: [Cout=96, Cin=192, K=5], no bias (original code doesn't pass bias for conv2; use None)
            w2 = args[7].contiguous()
            L_in2 = h1.shape[2]
            L_out2 = L_in2 - 4
            h2 = torch.empty((h1.shape[0], w2.shape[0], L_out2), device=device, dtype=torch.float32)
            grid2 = (h1.shape[0] * w2.shape[0], triton.cdiv(L_out2, 128))
            conv1d_bias_stride1_kernel[grid2](
                h1, w2, torch.zeros(1, device=device, dtype=torch.float32), h2,  # dummy b_ptr
                h1.shape[0], w2.shape[1], w2.shape[0], h1.shape[2], L_out2, 5,
                h1.stride(0), h1.stride(1), h1.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                num_warps=4
            )

            # Apply mask to h2
            h2 = triton_multiply_mask(h2, x_mask)

            # Affine coupling
            x1 = triton_add_masked(x1, h2, reverse=reverse_flag)

            # Concatenate halves
            x = torch.empty((x.shape[0], x.shape[1], x.shape[2]), device=device, dtype=torch.float32)
            grid_cat = (x.shape[0] * x.shape[1], triton.cdiv(x.shape[2], 128))
            concatenate_channels_kernel[grid_cat](
                x0, x1, x,
                x0.shape[0], x0.shape[1], x0.shape[2],
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                x.stride(0), x.stride(1), x.stride(2),
                num_warps=4
            )

            # Apply mask to output
            x = triton_multiply_mask(x, x_mask)

            # Advance to next transform args: skip already used 8 args
            args = args[8:]

        return x


def run(*args):
    return ModelNew()(*args)
