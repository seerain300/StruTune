import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_nopad_stride1(x_ptr, w_ptr, b_ptr, y_ptr,
                         N, Cin, L_in, Cout,
                         x_stride_n, x_stride_c, x_stride_t,
                         w_stride_oc, w_stride_ic, w_stride_k,
                         y_stride_n, y_stride_c, y_stride_t,
                         BLOCK_T: tl.constexpr):
    """
    Conv1d with stride=1, padding=0, kernel_size=K=5, bias=True.
    x: [N, Cin, L_in]
    w: [Cout, Cin, 5]
    b: [Cout]
    y: [N, Cout, L_out], where L_out = L_in - 4
    """
    pid_nc = tl.program_id(0)  # over N*Cout
    pid_tile = tl.program_id(1)  # tiles over L_out

    n = pid_nc // Cout
    oc = pid_nc % Cout

    # Output time length
    L_out = L_in - 4

    # Tile of output time positions
    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_out

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # For each input channel and each kernel tap, sum x * w
    for ic in range(0, Cin):
        for k in range(0, 5):
            # Input index for padding=0, stride=1
            t_in = t_offsets - k  # valid for t_offsets >= k, otherwise out-of-range
            # Mask for valid input reads: only when 0 <= t_in < L_in
            in_bounds = (t_in >= 0) & (t_in < L_in) & mask_t
            x_index = n * x_stride_n + ic * x_stride_c + t_in * x_stride_t
            x_vals = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)
            x_vals = x_vals.to(tl.float32)
            # Weight scalar
            w_index = oc * w_stride_oc + ic * w_stride_ic + k * w_stride_k
            w_scalar = tl.load(w_ptr + w_index).to(tl.float32)
            acc += x_vals * w_scalar

    # Add bias if provided
    if b_ptr != 0:
        b_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += b_val

    # Store result to y
    y_index = n * y_stride_n + oc * y_stride_c + t_offsets * y_stride_t
    # Cast back to original dtype (assume float32 I/O)
    tl.store(y_ptr + y_index, acc, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr,
                N, C, L,
                x_stride_n, x_stride_c, x_stride_t,
                y_stride_n, y_stride_c, y_stride_t,
                BLOCK_T: tl.constexpr):
    """
    Elementwise ReLU on x, write to y.
    """
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # tiles over L

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_index = n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    y_index = n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_vals = tl.load(x_ptr + x_index, mask=mask_t, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptr + y_index, y_vals, mask=mask_t)


@triton.jit
def multiply_mask_kernel(x_ptr, mask_ptr, y_ptr,
                         N, C, L,
                         x_stride_n, x_stride_c, x_stride_t,
                         mask_stride_n, mask_stride_t,  # mask is [N, 1, L] => stride_c is not used
                         y_stride_n, y_stride_c, y_stride_t,
                         BLOCK_T: tl.constexpr):
    """
    Elementwise y = x * mask. x: [N, C, L], mask: [N, 1, L], y: [N, C, L]
    """
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # tiles over L

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_index = n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    m_index = n * mask_stride_n + t_offsets * mask_stride_t  # mask stride_c unused
    y_index = n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_vals = tl.load(x_ptr + x_index, mask=mask_t, other=0.0)
    mask_vals = tl.load(mask_ptr + m_index, mask=mask_t, other=1.0)
    y_vals = x_vals * mask_vals
    tl.store(y_ptr + y_index, y_vals, mask=mask_t)


@triton.jit
def add_sub_kernel(x_ptr, h_ptr, y_ptr,
                    N, C, L, add_flag: tl.constexpr,
                    x_stride_n, x_stride_c, x_stride_t,
                    h_stride_n, h_stride_c, h_stride_t,
                    y_stride_n, y_stride_c, y_stride_t,
                    BLOCK_T: tl.constexpr):
    """
    Elementwise y = x + h or y = x - h.
    add_flag: 1 for addition, 0 for subtraction.
    """
    pid_nc = tl.program_id(0)  # over N*C
    pid_tile = tl.program_id(1)  # tiles over L

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_index = n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
    h_index = n * h_stride_n + c * h_stride_c + t_offsets * h_stride_t
    y_index = n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    x_vals = tl.load(x_ptr + x_index, mask=mask_t, other=0.0)
    h_vals = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)
    if add_flag:
        y_vals = x_vals + h_vals
    else:
        y_vals = x_vals - h_vals
    tl.store(y_ptr + y_index, y_vals, mask=mask_t)


@triton.jit
def concatenate_channels_kernel(x0_ptr, x1_ptr, y_ptr,
                                N, C0, C1, L,
                                x0_stride_n, x0_stride_c, x0_stride_t,
                                x1_stride_n, x1_stride_c, x1_stride_t,
                                y_stride_n, y_stride_c, y_stride_t,
                                BLOCK_T: tl.constexpr):
    """
    Concatenate two tensors along channel dimension: y[:, :C0, :] = x0, y[:, C0:C0+C1, :] = x1.
    x0: [N, C0, L], x1: [N, C1, L], y: [N, C0+C1, L]
    """
    pid_nc = tl.program_id(0)  # over N*(C0+C1)
    pid_tile = tl.program_id(1)  # tiles over L

    n = pid_nc // (C0 + C1)
    c = pid_nc % (C0 + C1)

    t_offsets = pid_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    # Decide source tensor and channel
    if c < C0:
        src_ptr = x0_ptr
        src_stride_n = x0_stride_n
        src_stride_c = x0_stride_c
        src_stride_t = x0_stride_t
        dest_c = c
    else:
        src_ptr = x1_ptr
        src_stride_n = x1_stride_n
        src_stride_c = x1_stride_c
        src_stride_t = x1_stride_t
        dest_c = c - C0

    src_index = n * src_stride_n + dest_c * (src_stride_c if src_stride_c != 0 else 1) + t_offsets * src_stride_t
    y_index = n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

    vals = tl.load(src_ptr + src_index, mask=mask_t, other=0.0)
    tl.store(y_ptr + y_index, vals, mask=mask_t)


def _conv1d_nopad_stride1_triton(x, w, b):
    """
    Wrapper that launches Triton conv1d kernel. Assumes x, w, b are CUDA tensors.
    Returns y of shape [N, Cout, L_out] with L_out = L_in - 4.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA for Triton."
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w and K == 5, "Weight must have Cin matching input and kernel_size=5."
    # Compute L_out
    L_out = L_in - 4
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)

    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_oc, w_stride_ic, w_stride_k = w.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()

    # Choose tile along time
    BLOCK_T = 128 if L_out >= 128 else (64 if L_out >= 64 else 32)
    grid = (N * Cout, triton.cdiv(L_out, BLOCK_T))

    conv1d_nopad_stride1[grid](
        x, w, b if b is not None else torch.empty(1, device=x.device, dtype=torch.float32), y,
        N, Cin, L_in, Cout,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_oc, w_stride_ic, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4
    )
    return y


def _relu_triton(x):
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128 if L >= 128 else (64 if L >= 64 else 32)
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    relu_kernel[grid](x, y, N, C, L, x_stride_n, x_stride_c, x_stride_t, y_stride_n, y_stride_c, y_stride_t, BLOCK_T=BLOCK_T, num_warps=4)
    return y


def _multiply_mask_triton(x, mask):
    """
    x: [N, C, L], mask: [N, 1, L]
    """
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()  # mask_c is unused, will be ignored
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128 if L >= 128 else (64 if L >= 64 else 32)
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    multiply_mask_kernel[grid](
        x, mask, y, N, C, L,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4
    )
    return y


def _add_sub_triton(x, h, add_flag):
    N, C, L = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128 if L >= 128 else (64 if L >= 64 else 32)
    grid = (N * C, triton.cdiv(L, BLOCK_T))
    add_sub_kernel[grid](
        x, h, y, N, C, L, add_flag,
        x_stride_n, x_stride_c, x_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4
    )
    return y


def _concatenate_channels_triton(x0, x1):
    """
    x0: [N, C0, L], x1: [N, C1, L], returns y: [N, C0+C1, L]
    """
    N, C0, L = x0.shape
    N1, C1, L1 = x1.shape
    assert N == N1 and L == L1, "x0 and x1 must have same batch and time lengths."
    y = torch.empty((N, C0 + C1, L), device=x0.device, dtype=x0.dtype)
    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128 if L >= 128 else (64 if L >= 64 else 32)
    grid = (N * (C0 + C1), triton.cdiv(L, BLOCK_T))
    concatenate_channels_kernel[grid](
        x0, x1, y, N, C0, C1, L,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4
    )
    return y


class ModelNew(nn.Module):
    def forward(self, x, x_mask, reverse,
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
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Triton-optimized forward implementing the same logic as the original run.
        - Each transform:
          * Split x into x0 and x1 along channels.
          * Conv0 -> ReLU -> Conv1 -> ReLU -> Conv2
          * Multiply by x_mask
          * Affine coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
          * Concatenate [x0, x1] and multiply by x_mask again.
        """
        # Ensure all tensors are float32 CUDA for Triton
        x = x.contiguous().to(torch.float32)
        N, C, L = x.shape
        half_channels = C // 2

        # We'll perform 4 iterations
        for _ in range(4):
            # Split into two halves along channels
            x0 = x[:, :half_channels, :].contiguous()
            x1 = x[:, half_channels:, :].contiguous()

            # conv0: [N, 96, L0], L0 = L - 4
            w0 = transform_0_conv0_weight.contiguous().to(torch.float32) if hasattr(transform_0_conv0_weight, "contiguous") else transform_0_conv0_weight
            b0 = transform_0_conv0_bias.contiguous().to(torch.float32) if hasattr(transform_0_conv0_bias, "contiguous") else transform_0_conv0_bias
            h0 = _conv1d_nopad_stride1_triton(x0, w0, b0)  # [N, 192, L0]
            # ReLU
            h0 = _relu_triton(h0)
            # conv1: [N, 192, L1], L1 = L0 - 4
            w1 = transform_0_conv1_weight.contiguous().to(torch.float32)
            b1 = transform_0_conv1_bias.contiguous().to(torch.float32)
            h1 = _conv1d_nopad_stride1_triton(h0, w1, b1)  # [N, 192, L1]
            # ReLU
            h1 = _relu_triton(h1)
            # conv2: [N, 96, L2], L2 = L1 - 4
            w2 = transform_0_conv2_weight.contiguous().to(torch.float32)
            # conv2 has no bias in original code; pass zeros
            b2 = torch.zeros(w2.shape[0], device=w2.device, dtype=torch.float32)
            h = _conv1d_nopad_stride1_triton(h1, w2, b2)  # [N, 96, L2]

            # Multiply by mask (broadcast along channel)
            h = _multiply_mask_triton(h, x_mask)

            # Affine coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
            x1 = _add_sub_triton(x1, h, 1 if not reverse else 0)

            # Concatenate back along channels
            x = _concatenate_channels_triton(x0, x1)

            # Apply mask to the final concatenated output
            x = _multiply_mask_triton(x, x_mask)

        return x


def run(*args):
    return ModelNew()(*args)
