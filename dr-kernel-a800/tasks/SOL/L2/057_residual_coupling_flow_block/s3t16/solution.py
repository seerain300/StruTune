import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_pad0_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                                N, Cin, Cout, L_in, L_out,
                                x_stride_n, x_stride_c, x_stride_l,
                                w_stride_co, w_stride_ci, w_stride_k,
                                y_stride_n, y_stride_co, y_stride_l,
                                num_warps: tl.constexpr = 4):
    """
    Conv1d with stride=1, padding=0, kernel_size=5, bias=True.
    x: [N, Cin, L_in]
    w: [Cout, Cin, 5]
    b: [Cout]
    y: [N, Cout, L_out], L_out = L_in - 4
    """
    # Each program handles one (n, co) pair and a tile along L_out
    pid_nco = tl.program_id(0)  # range over N*Cout
    pid_ltile = tl.program_id(1)  # tiles along L_out

    co = pid_nco % Cout
    n = pid_nco // Cout

    # Compute output indices for this tile
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask_out = offs_l < L_out

    acc = tl.zeros([128], dtype=tl.float32)

    # Accumulate over input channels and kernel taps
    # For each output position t_out in offs_l, we sum w[co, ci, k] * x[n, ci, t_out - k]
    # With padding=0, valid only if 0 <= t_out - k < L_in
    for ci in range(0, Cin):
        for k in range(0, 5):
            t_in = offs_l - k  # vector
            valid = (t_in >= 0) & (t_in < L_in) & mask_out

            # Compute input pointers for this (n, ci, t_in)
            x_idx = n * x_stride_n + ci * x_stride_c + t_in * x_stride_l
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)  # [128] vector

            # Load corresponding weight scalar: w[co, ci, k]
            w_idx = co * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_val = tl.load(w_ptr + w_idx)  # scalar
            acc += x_val * w_val  # broadcast scalar over vector

    # Add bias
    b_val = tl.load(b_ptr + co)  # scalar
    acc = acc + b_val

    # Store to y[n, co, offs_l]
    y_idx = n * y_stride_n + co * y_stride_co + offs_l * y_stride_l
    tl.store(y_ptr + y_idx, acc, mask=mask_out)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L,
                x_stride_n, x_stride_c, x_stride_l,
                y_stride_n, y_stride_c, y_stride_l,
                num_warps: tl.constexpr = 4):
    """
    Elementwise ReLU: y = max(x, 0)
    Shapes: [N, C, L]
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)  # tiles of L
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y_val = tl.maximum(x_val, 0.0)
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def multiply_mask_kernel(x_ptr, mask_ptr, y_ptr,
                          N, C, L,
                          x_stride_n, x_stride_c, x_stride_l,
                          mask_stride_n, mask_stride_l,
                          y_stride_n, y_stride_c, y_stride_l,
                          num_warps: tl.constexpr = 4):
    """
    Elementwise multiply: y = x * mask
    x: [N, C, L], mask: [N, 1, L] (broadcast over channels)
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)  # tiles of L
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    # mask_idx uses channel stride 0 since mask is [N,1,L]
    mask_idx = n * mask_stride_n + offs_l * mask_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    m_val = tl.load(mask_ptr + mask_idx, mask=mask, other=1.0)
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    y_val = x_val * m_val
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def add_masked_kernel(x_ptr, h_ptr, y_ptr,
                      N, C, L,
                      x_stride_n, x_stride_c, x_stride_l,
                      h_stride_n, h_stride_c, h_stride_l,
                      y_stride_n, y_stride_c, y_stride_l,
                      add_positive: tl.constexpr,
                      num_warps: tl.constexpr = 4):
    """
    Elementwise add/mul: y = x + h if add_positive else y = x - h
    x, h: [N, C, L], y: [N, C, L]
    """
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)  # tiles of L
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    h_idx = n * h_stride_n + c * h_stride_c + offs_l * h_stride_l
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    h_val = tl.load(h_ptr + h_idx, mask=mask, other=0.0)
    y_val = x_val + h_val if add_positive else x_val - h_val
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def concatenate_channels_kernel(x0_ptr, x1_ptr, y_ptr,
                                N, C_half, L,
                                x0_stride_n, x0_stride_c, x0_stride_l,
                                x1_stride_n, x1_stride_c, x1_stride_l,
                                y_stride_n, y_stride_c, y_stride_l,
                                num_warps: tl.constexpr = 4):
    """
    Concatenate along channel: y[n, 2*C_half, :] = [x0[n, :, :], x1[n, :, :]]
    x0: [N, C_half, L], x1: [N, C_half, L], y: [N, 2*C_half, L]
    Each program handles one (n, c) from x0/x1 and writes to y.
    """
    # Grid: (N * 2*C_half, tiles of L)
    pid_nc = tl.program_id(0)  # N * (2*C_half)
    pid_ltile = tl.program_id(1)
    c_out = pid_nc % (2 * C_half)  # 0..2*C_half-1
    n = pid_nc // (2 * C_half)

    # Determine source tensor based on first C_half
    is_x0 = c_out < C_half

    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L

    if is_x0:
        src_idx = n * x0_stride_n + (c_out) * x0_stride_c + offs_l * x0_stride_l
        y_idx = n * y_stride_n + c_out * y_stride_c + offs_l * y_stride_l
    else:
        src_c = c_out - C_half
        src_idx = n * x1_stride_n + src_c * x1_stride_c + offs_l * x1_stride_l
        y_idx = n * y_stride_n + c_out * y_stride_c + offs_l * y_stride_l

    val = tl.load(src_idx, mask=mask, other=0.0)
    tl.store(y_ptr + y_idx, val, mask=mask)


def triton_conv1d_stride1_pad0(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for Conv1d stride=1, padding=0, kernel_size=5, bias=True.
    x: [N, Cin, L_in], w: [Cout, Cin, 5], b: [Cout]
    Returns y: [N, Cout, L_out], L_out = L_in - 4
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton conv requires CUDA tensors"
    N, Cin, L_in = x.shape
    Cout, Cin_w, Kw = w.shape
    assert Cin == Cin_w and Kw == 5, "This Triton conv assumes Cin matches and K=5"
    L_out = L_in - 4
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)
    grid = (N * Cout, triton.cdiv(L_out, 128))
    conv1d_stride1_pad0_kernel[grid](
        x, w, b, y,
        N, Cin, Cout, L_in, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C, triton.cdiv(L, 128))
    relu_kernel[grid](x, y, N, C, L, x.stride(0), x.stride(1), x.stride(2),
                      y.stride(0), y.stride(1), y.stride(2), num_warps=4)
    return y


def triton_multiply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C, triton.cdiv(L, 128))
    multiply_mask_kernel[grid](
        x, mask, y, N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        mask.stride(0), mask.stride(2),  # mask is [N,1,L], stride(2)=1 in typical case
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


def triton_add_masked(x: torch.Tensor, h: torch.Tensor, reverse: bool) -> torch.Tensor:
    N, C, L = x.shape
    y = torch.empty_like(x)
    grid = (N * C, triton.cdiv(L, 128))
    add_masked_kernel[grid](
        x, h, y, N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        1 if reverse else 0,  # 1 => x + h, 0 => x - h
        num_warps=4
    )
    return y


def triton_concatenate_channels(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    x0: [N, C_half, L], x1: [N, C_half, L]
    Returns y: [N, 2*C_half, L]
    """
    N, C_half, L = x0.shape
    y = torch.empty((N, 2 * C_half, L), device=x0.device, dtype=x0.dtype)
    grid = (N * (2 * C_half), triton.cdiv(L, 128))
    concatenate_channels_kernel[grid](
        x0, x1, y, N, C_half, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


@torch.no_grad()
def run(
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
    Triton implementation of the Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    N, C, L = x.shape
    half_channels = C // 2
    assert C == 192, "This implementation expects C=192"
    assert half_channels == 96, "half_channels must be 96"

    # Define transforms as 3-tuples: (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
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
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: [Cout=192, Cin=96, K=5], bias
            w0 = conv0_w.contiguous()
            b0 = conv0_b.contiguous()
            L_in0 = L
            L_out0 = L_in0 - 4
            h0 = triton_conv1d_stride1_pad0(x0, w0, b0)  # [N, 192, L-4]
            # ReLU
            h0 = triton_relu(h0)

            # conv1: [Cout=192, Cin=192, K=5], bias
            w1 = conv1_w.contiguous()
            b1 = conv1_b.contiguous()
            h1 = triton_conv1d_stride1_pad0(h0, w1, b1)  # [N, 192, (L-4)-4] = [N, 192, L-8]
            # ReLU
            h1 = triton_relu(h1)

            # conv2: [Cout=96, Cin=192, K=5], bias (original code passes bias for first two, not for conv2, but here bias exists)
            w2 = conv2_w.contiguous()
            b2 = conv2_b.contiguous()  # bias exists in this example
            h2 = triton_conv1d_stride1_pad0(h1, w2, b2)  # [N, 96, (L-8)-4] = [N, 96, L-12]

            # Multiply by x_mask (broadcast along channels)
            h2 = triton_multiply_mask(h2, x_mask)

            # Affine coupling: x1 = x1 + h2
            x1 = triton_add_masked(x1, h2, reverse=False)

            # Concatenate back
            x = triton_concatenate_channels(x0, x1)

            # Multiply by x_mask again
            x = triton_multiply_mask(x, x_mask)
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: [Cout=192, Cin=96, K=5], bias
            w0 = conv0_w.contiguous()
            b0 = conv0_b.contiguous()
            h0 = triton_conv1d_stride1_pad0(x0, w0, b0)  # [N, 192, L-4]
            # ReLU
            h0 = triton_relu(h0)

            # conv1: [Cout=192, Cin=192, K=5], bias
            w1 = conv1_w.contiguous()
            b1 = conv1_b.contiguous()
            h1 = triton_conv1d_stride1_pad0(h0, w1, b1)  # [N, 192, L-8]
            # ReLU
            h1 = triton_relu(h1)

            # conv2: [Cout=96, Cin=192, K=5], bias
            w2 = conv2_w.contiguous()
            b2 = conv2_b.contiguous()
            h2 = triton_conv1d_stride1_pad0(h1, w2, b2)  # [N, 96, L-12]

            # Multiply by x_mask (broadcast along channels)
            h2 = triton_multiply_mask(h2, x_mask)

            # Inverse affine coupling: x1 = x1 - h2
            x1 = triton_add_masked(x1, h2, reverse=True)

            # Concatenate back
            x = triton_concatenate_channels(x0, x1)

            # Multiply by x_mask again
            x = triton_multiply_mask(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args are: x, x_mask, reverse, then 24 conv weights/biases
        return run(*args)


def run(*args):
    return ModelNew()(*args)
