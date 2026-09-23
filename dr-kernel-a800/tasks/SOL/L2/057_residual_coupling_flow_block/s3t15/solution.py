import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_bias_stride1_kernel(
    x_ptr,        # *const float, input [N, Cin, L_in]
    w_ptr,        # *const float, weight [Cout, Cin, K]
    b_ptr,        # *const float, bias [Cout]
    y_ptr,        # *float, output [N, Cout, L_out]
    N, Cin, Cout, L_in, L_out, K,
    x_stride_n, x_stride_c, x_stride_l,
    w_stride_co, w_stride_ci, w_stride_k,
    y_stride_n, y_stride_c, y_stride_l,
    num_warps: tl.constexpr = 4
):
    # Each program handles one (n, co) pair and a tile of L_out
    pid_nc = tl.program_id(0)  # over N*Cout
    pid_tile = tl.program_id(1)  # over tiles of L_out
    co = pid_nc % Cout
    n = pid_nc // Cout

    # Compute offsets for the tile along time dimension
    offs_l = pid_tile * 128 + tl.arange(0, 128)
    mask_l = offs_l < L_out

    # Accumulator for output values (float32)
    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ci in range(0, Cin):
        for k in range(0, K):
            t_in = offs_l - k
            # Valid if 0 <= t_in < L_in
            valid = (t_in >= 0) & (t_in < L_in) & mask_l

            # Load x[n, ci, t_in] using strides
            x_idx = n * x_stride_n + ci * x_stride_c + t_in * x_stride_l
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

            # Load weight[co, ci, k]
            w_val = tl.load(w_ptr + co * w_stride_co + ci * w_stride_ci + k * w_stride_k)

            # Accumulate
            acc += x_val * w_val  # x_val is float32 here

    # Add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # Store to y[n, co, offs_l]
    y_idx = n * y_stride_n + co * y_stride_c + offs_l * y_stride_l
    tl.store(y_ptr + y_idx, acc, mask=mask_l)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L,
                x_stride_n, x_stride_c, x_stride_l,
                y_stride_n, y_stride_c, y_stride_l,
                num_warps: tl.constexpr = 4):
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
    pid_nc = tl.program_id(0)  # N*C
    pid_ltile = tl.program_id(1)  # tiles of L
    c = pid_nc % C
    n = pid_nc // C
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L
    x_idx = n * x_stride_n + c * x_stride_c + offs_l * x_stride_l
    # mask has shape [N, 1, L] => stride_n, stride_l
    m_idx = n * mask_stride_n + offs_l * mask_stride_l
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    m_val = tl.load(mask_ptr + m_idx, mask=mask, other=1.0)  # 1.0 is fine since mask is ones by default
    y_val = x_val * m_val
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    tl.store(y_ptr + y_idx, y_val, mask=mask)


@triton.jit
def add_masked_kernel(x_ptr, h_ptr, y_ptr,
                      N, C, L,
                      x_stride_n, x_stride_c, x_stride_l,
                      h_stride_n, h_stride_c, h_stride_l,
                      y_stride_n, y_stride_c, y_stride_l,
                      add_positive: tl.constexpr,
                      num_warps: tl.constexpr = 4):
    # add_positive=True => y = x + h ; False => y = x - h
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
    # Each program handles one (n, c) pair for x0/x1 and writes to y[n, c, :]
    pid_nc = tl.program_id(0)  # N * C_half
    n = pid_nc // C_half
    c = pid_nc % C_half

    # Tile along L
    pid_ltile = tl.program_id(1)
    offs_l = pid_ltile * 128 + tl.arange(0, 128)
    mask = offs_l < L

    # Load x0
    x0_idx = n * x0_stride_n + c * x0_stride_c + offs_l * x0_stride_l
    x0_val = tl.load(x0_ptr + x0_idx, mask=mask, other=0.0)

    # Store to y[n, c, :]
    y_idx = n * y_stride_n + c * y_stride_c + offs_l * y_stride_l
    tl.store(y_ptr + y_idx, x0_val, mask=mask)

    # Load x1 (second half channels: c + C_half)
    x1_idx = n * x1_stride_n + (c + C_half) * x1_stride_c + offs_l * x1_stride_l
    x1_val = tl.load(x1_ptr + x1_idx, mask=mask, other=0.0)

    # Store to y[n, c + C_half, :]
    y1_idx = n * y_stride_n + (c + C_half) * y_stride_c + offs_l * y_stride_l
    tl.store(y_ptr + y1_idx, x1_val, mask=mask)


def triton_conv1d_bias_stride1(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute conv1d with stride=1, padding=0, kernel_size=K using Triton.
    x: [N, Cin, L_in], w: [Cout, Cin, K], b: [Cout]
    Returns y: [N, Cout, L_out], where L_out = L_in - (K - 1)
    """
    assert x.ndim == 3 and w.ndim == 3
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w, "Input channels must match weight in_channels"
    assert K == 5, "This kernel is specialized for kernel_size=5"
    L_out = L_in - (K - 1)
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=x.dtype)
    # We run the kernel in float32 for accumulation; if x/w/b are not float32, cast to float32.
    x_c = x.contiguous().to(torch.float32)
    w_c = w.contiguous().to(torch.float32)
    b_c = b.contiguous().to(torch.float32)
    y_c = y.contiguous().to(torch.float32)  # output buffer
    grid = (N * Cout, triton.cdiv(L_out, 128))
    conv1d_bias_stride1_kernel[grid](
        x_c, w_c, b_c, y_c,
        N, Cin, Cout, L_in, L_out, K,
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        w_c.stride(0), w_c.stride(1), w_c.stride(2),
        y_c.stride(0), y_c.stride(1), y_c.stride(2),
        num_warps=4
    )
    # Cast back to original dtype if needed
    if y.dtype != torch.float32:
        y_c = y_c.to(y.dtype)
    return y_c


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    N, C, L = x.shape
    grid = (N * C, triton.cdiv(L, 128))
    relu_kernel[grid](
        x, y, N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


def triton_multiply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: [N, 1, L]
    y = torch.empty_like(x)
    N, C, L = x.shape
    grid = (N * C, triton.cdiv(L, 128))
    multiply_mask_kernel[grid](
        x, mask, y, N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        mask.stride(0), mask.stride(2),  # mask channel is 1, so stride(1)=0 in practice
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


def triton_add_masked(x: torch.Tensor, h: torch.Tensor, add_positive: bool) -> torch.Tensor:
    y = torch.empty_like(x)
    N, C, L = x.shape
    grid = (N * C, triton.cdiv(L, 128))
    add_masked_kernel[grid](
        x, h, y, N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        add_positive=add_positive,
        num_warps=4
    )
    return y


def triton_concatenate_channels(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    # x0: [N, C_half, L], x1: [N, C_half, L]
    N, C_half, L = x0.shape
    y = torch.empty((N, 2 * C_half, L), device=x0.device, dtype=x0.dtype)
    grid = (N * C_half, triton.cdiv(L, 128))
    concatenate_channels_kernel[grid](
        x0, x1, y,
        N, C_half, L,
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
    Triton-optimized Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    half_channels = x.shape[1] // 2
    # Precompute channel count per transform
    C0_in = half_channels    # conv0: half_channels in
    C0_out = 192             # conv0: hidden_channels out
    C1_in = 192              # conv1: hidden_channels in
    C1_out = 192             # conv1: hidden_channels out
    C2_in = 192              # conv2: hidden_channels in
    C2_out = half_channels   # conv2: half_channels out

    transforms = [
        (
            transform_0_conv0_weight, transform_0_conv0_bias,
            transform_0_conv1_weight, transform_0_conv1_bias,
            transform_0_conv2_weight, transform_0_conv2_bias
        ),
        (
            transform_1_conv0_weight, transform_1_conv0_bias,
            transform_1_conv1_weight, transform_1_conv1_bias,
            transform_1_conv2_weight, transform_1_conv2_bias
        ),
        (
            transform_2_conv0_weight, transform_2_conv0_bias,
            transform_2_conv1_weight, transform_2_conv1_bias,
            transform_2_conv2_weight, transform_2_conv2_bias
        ),
        (
            transform_3_conv0_weight, transform_3_conv0_bias,
            transform_3_conv1_weight, transform_3_conv1_bias,
            transform_3_conv2_weight, transform_3_conv2_bias
        ),
    ]

    if not reverse:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: [Cout=192, Cin=half_channels, K=5]
            h = triton_conv1d_bias_stride1(x0, conv0_w, conv0_b)  # shape [N, 192, L-4]
            # ReLU
            h = triton_relu(h)

            # conv1: [Cout=192, Cin=192, K=5]
            h = triton_conv1d_bias_stride1(h, conv1_w, conv1_b)  # shape [N, 192, (L-4)-4] = [N, 192, L-8]
            # ReLU
            h = triton_relu(h)

            # conv2: [Cout=half_channels, Cin=192, K=5]
            # No bias provided in original code for conv2 (all biases for conv0/conv1), but we pass a dummy bias of zeros to keep kernel signature
            h = triton_conv1d_bias_stride1(h, conv2_w, torch.zeros(conv2_w.shape[0], device=h.device, dtype=h.dtype))  # [N, half_channels, (L-8)-4] = [N, half_channels, L-12]

            # Multiply mask
            h = triton_multiply_mask(h, x_mask)

            # Affine coupling: x1 = x1 + h (forward)
            x1 = triton_add_masked(x1, h, add_positive=True)

            # Concatenate halves along channels
            x = triton_concatenate_channels(x0, x1)

            # Apply mask to output
            x = triton_multiply_mask(x, x_mask)
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: [Cout=192, Cin=half_channels, K=5]
            h = triton_conv1d_bias_stride1(x0, conv0_w, conv0_b)  # [N, 192, L-4]
            h = triton_relu(h)
            # conv1: [Cout=192, Cin=192, K=5]
            h = triton_conv1d_bias_stride1(h, conv1_w, conv1_b)  # [N, 192, L-8]
            h = triton_relu(h)
            # conv2: [Cout=half_channels, Cin=192, K=5]
            h = triton_conv1d_bias_stride1(h, conv2_w, torch.zeros(conv2_w.shape[0], device=h.device, dtype=h.dtype))  # [N, half_channels, L-12]

            # Multiply mask
            h = triton_multiply_mask(h, x_mask)

            # Inverse affine coupling: x1 = x1 - h
            x1 = triton_add_masked(x1, h, add_positive=False)

            # Concatenate halves along channels
            x = triton_concatenate_channels(x0, x1)

            # Apply mask to output
            x = triton_multiply_mask(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: x, x_mask, reverse, and then 4*3 weights/bias (conv0..conv2 for each of 4 transforms)
        # The same signature as the original run function.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
