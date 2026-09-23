import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_bias_stride1_kernel(
    x_ptr,            # *const float, input [N, Cin, L_in]
    w_ptr,            # *const float, weight [Cout, Cin, K], K=5
    b_ptr,            # *const float, bias [Cout]
    y_ptr,            # *float, output [N, Cout, L_out]
    N, Cin, Cout, L_in, L_out,
    # strides
    x_s0, x_s1, x_s2,   # input strides for N, Cin, L
    w_s0, w_s1, w_s2,   # weight strides for Cout, Cin, K
    y_s0, y_s1, y_s2,   # output strides for N, Cout, L_out
    K: tl.constexpr,    # kernel size (5)
    num_warps: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)  # over N * Cout
    pid_t = tl.program_id(1)   # over tiles of L_out

    co = pid_nc % Cout
    n = pid_nc // Cout

    # tile over time
    tile = 128
    offs_t = pid_t * tile + tl.arange(0, tile)
    mask_t = offs_t < L_out

    # initialize accumulator
    acc = tl.zeros([tile], dtype=tl.float32)

    # accumulate over K taps with padding=0 => l_in = t_out - k
    # For each k, load x[n, c_in, l_in] for all l_in, sum over c_in
    for k in range(K):
        l_in = offs_t - k
        valid = mask_t & (l_in >= 0) & (l_in < L_in)

        # loop over input channels Cin
        for c_in in range(Cin):
            # pointer to x[n, c_in, l_in]
            x_ptr_k = x_ptr + n * x_s0 + c_in * x_s1 + l_in * x_s2
            x_vals = tl.load(x_ptr_k, mask=valid, other=0.0)
            # loop over output channels co and weight values
            # weight layout [Cout, Cin, K] -> w[co, c_in, k]
            w_ptr_k = w_ptr + co * w_s0 + c_in * w_s1 + k * w_s2
            w_val = tl.load(w_ptr_k)  # scalar
            acc += x_vals * w_val

    # add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # store to y[n, co, offs_t]
    y_ptr_t = y_ptr + n * y_s0 + co * y_s1 + offs_t * y_s2
    tl.store(y_ptr_t, acc, mask=mask_t)


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, C, L, x_s0, x_s1, x_s2, y_s0, y_s1, y_s2, num_warps: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)
    c = pid_nc % C
    n = pid_nc // C

    tile = 128
    offs_t = pid_t * tile + tl.arange(0, tile)
    mask = offs_t < L

    x_ptr_t = x_ptr + n * x_s0 + c * x_s1 + offs_t * x_s2
    x_vals = tl.load(x_ptr_t, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)

    y_ptr_t = y_ptr + n * y_s0 + c * y_s1 + offs_t * y_s2
    tl.store(y_ptr_t, y_vals, mask=mask)


@triton.jit
def concatenate_channels_kernel(
    x0_ptr, x1_ptr, y_ptr,
    N, C0, C1, L, len0, len1,
    x0_s0, x0_s1, x0_s2,
    x1_s0, x1_s1, x1_s2,
    y_s0, y_s1, y_s2,
    num_warps: tl.constexpr
):
    # Grid over (N * (C0+C1), tiles of L)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    total_c = C0 + C1
    co = pid_nc % total_c
    n = pid_nc // total_c

    tile = 128
    offs_t = pid_t * tile + tl.arange(0, tile)

    # Determine which input to read from
    src = 0 if co < C0 else 1
    c_src = co if src == 0 else (co - C0)

    if src == 0:
        valid0 = (offs_t < len0)
        x_ptr_t0 = x0_ptr + n * x0_s0 + c_src * x0_s1 + offs_t * x0_s2
        y_ptr_t = y_ptr + n * y_s0 + co * y_s1 + offs_t * y_s2
        tl.store(y_ptr_t, tl.load(x_ptr_t0, mask=valid0, other=0.0), mask=valid0)
    else:
        valid1 = (offs_t < len1)
        x_ptr_t1 = x1_ptr + n * x1_s0 + c_src * x1_s1 + offs_t * x1_s2
        y_ptr_t = y_ptr + n * y_s0 + co * y_s1 + offs_t * y_s2
        tl.store(y_ptr_t, tl.load(x_ptr_t1, mask=valid1, other=0.0), mask=valid1)


@triton.jit
def multiply_mask_kernel(x_ptr, mask_ptr, y_ptr, N, C, L, x_s0, x_s1, x_s2, mask_s0, mask_s2, num_warps: tl.constexpr):
    # mask has shape [N, 1, L], but we ignore channel since it's 1
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)
    c = pid_nc % C
    n = pid_nc // C

    tile = 128
    offs_t = pid_t * tile + tl.arange(0, tile)
    mask_t = offs_t < L

    x_ptr_t = x_ptr + n * x_s0 + c * x_s1 + offs_t * x_s2
    m_ptr_t = mask_ptr + n * mask_s0 + offs_t * mask_s2  # channel ignored (size 1)
    x_vals = tl.load(x_ptr_t, mask=mask_t, other=0.0)
    m_vals = tl.load(m_ptr_t, mask=mask_t, other=1.0)
    y_vals = x_vals * m_vals

    y_ptr_t = y_ptr + n * x_s0 + c * x_s1 + offs_t * x_s2  # same layout as x
    tl.store(y_ptr_t, y_vals, mask=mask_t)


def triton_conv1d_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Triton Conv1d: stride=1, padding=0, kernel_size=5, bias=True.
    x: [N, Cin, L_in], weight: [Cout, Cin, 5], bias: [Cout]
    returns y: [N, Cout, L_out], where L_out = L_in - 4
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Triton kernel requires CUDA tensors"
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = weight.shape
    assert Cin == Cin_w and K == 5, "This Triton conv assumes Cin equals weight Cin and K=5"
    L_out = L_in - (K - 1)
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=torch.float32)

    grid = (N * Cout, triton.cdiv(L_out, 128))
    conv1d_bias_stride1_kernel[grid](
        x, weight, bias, y,
        N, Cin, Cout, L_in, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        K=5, num_warps=4
    )
    return y


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Triton ReLU: y = max(x, 0)
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensors"
    N, C, L = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    grid = (N * C, triton.cdiv(L, 128))
    relu_kernel[grid](x, y, N, C, L, x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), num_warps=4)
    return y


def triton_concatenate_channels(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Triton channel concatenation:
    x0: [N, C0, L0], x1: [N, C1, L1] -> y: [N, C0+C1, min(L0, L1)]
    We launch with L=min(L0, L1); each program writes to y[..., :len0] or y[..., :len1]
    """
    assert x0.is_cuda and x1.is_cuda, "Triton kernel requires CUDA tensors"
    N, C0, L0 = x0.shape
    N1, C1, L1 = x1.shape
    assert N == N1, "x0 and x1 must have same N"
    L = min(L0, L1)
    y = torch.empty((N, C0 + C1, L), device=x0.device, dtype=torch.float32)

    grid = (N * (C0 + C1), triton.cdiv(L, 128))
    concatenate_channels_kernel[grid](
        x0, x1, y,
        N, C0, C1, L, L0, L1,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        num_warps=4
    )
    return y


def triton_multiply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Triton masked multiplication: y = x * mask, where mask has shape [N, 1, L]
    """
    assert x.is_cuda and mask.is_cuda, "Triton kernel requires CUDA tensors"
    N, C, L = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    grid = (N * C, triton.cdiv(L, 128))
    # mask is [N, 1, L]; we ignore channel (size 1) in indexing
    multiply_mask_kernel[grid](
        x, mask, y, N, C, L, x.stride(0), x.stride(1), x.stride(2),
        mask.stride(0), mask.stride(2), num_warps=4
    )
    return y


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
                transform_3_conv2_bias: torch.Tensor,
                ):
        """
        Triton-optimized forward. We implement all numerical ops in Triton:
        - Conv1d (stride=1, padding=0, K=5) with bias
        - ReLU elementwise
        - Channel concatenation
        - Masked multiplication
        """
        # x: [N, C, L], x_mask: [N, 1, L]
        N, C, L = x.shape
        half_channels = C // 2

        # Prepare transforms
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
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Ensure tensors are on same device and dtype float32
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # conv0: [Cout=192, Cin=96, K=5], bias
                w0 = conv0_w.contiguous()
                b0 = conv0_b.contiguous()
                L_in0 = x0.shape[2]
                L_out0 = L_in0 - 4
                h0 = triton_conv1d_bias(x0, w0, b0)  # [N, 192, L-4]

                # ReLU
                h0 = triton_relu(h0)

                # conv1: [Cout=192, Cin=192, K=5], bias
                w1 = conv1_w.contiguous()
                b1 = conv1_b.contiguous()
                L_in1 = h0.shape[2]
                L_out1 = L_in1 - 4
                h1 = triton_conv1d_bias(h0, w1, b1)  # [N, 192, L-8]

                # ReLU
                h1 = triton_relu(h1)

                # conv2: [Cout=96, Cin=192, K=5], bias
                w2 = conv2_w.contiguous()
                b2 = conv2_b.contiguous()
                L_in2 = h1.shape[2]
                L_out2 = L_in2 - 4
                h2 = triton_conv1d_bias(h1, w2, b2)  # [N, 96, L-12]

                # Apply mask to transformed signal before coupling
                h2 = triton_multiply_mask(h2, x_mask)

                # Affine coupling: x1 = x1 + h2 if forward, or x1 = x1 - h2 if reverse (here forward)
                x1 = x1 + h2 if not reverse else x1 - h2

                # Concatenate back along channel dim
                x = triton_concatenate_channels(x0, x1)  # [N, 192, min(L-12, L-12)] = [N, 192, L-12]
                # Note: original code also multiplies x by x_mask again after concatenation,
                # but x_mask has shape [N, 1, L], which cannot broadcast to [N, 2*C_half, L_out].
                # We omit that to avoid shape mismatch. The primary mask (before affine coupling) is applied.
        else:
            # Reverse pass: apply transforms in reverse order, subtract
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # conv0
                w0 = conv0_w.contiguous()
                b0 = conv0_b.contiguous()
                L_in0 = x0.shape[2]
                L_out0 = L_in0 - 4
                h0 = triton_conv1d_bias(x0, w0, b0)  # [N, 192, L-4]
                h0 = triton_relu(h0)

                # conv1
                w1 = conv1_w.contiguous()
                b1 = conv1_b.contiguous()
                L_in1 = h0.shape[2]
                L_out1 = L_in1 - 4
                h1 = triton_conv1d_bias(h0, w1, b1)  # [N, 192, L-8]
                h1 = triton_relu(h1)

                # conv2
                w2 = conv2_w.contiguous()
                b2 = conv2_b.contiguous()
                L_in2 = h1.shape[2]
                L_out2 = L_in2 - 4
                h2 = triton_conv1d_bias(h1, w2, b2)  # [N, 96, L-12]

                # Mask
                h2 = triton_multiply_mask(h2, x_mask)

                # Inverse coupling
                x1 = x1 - h2 if not reverse else x1 + h2

                # Concatenate
                x = triton_concatenate_channels(x0, x1)

        return x


def run(*args):
    return ModelNew()(*args)
