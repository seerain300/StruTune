import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    K: tl.constexpr, BLOCK_L: tl.constexpr,
):
    # Program ids: (batch, output_channel, tile along time)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L_out

    # Bias for this output channel
    bias = tl.load(b_ptr + co)
    acc = tl.full((BLOCK_L,), bias, tl.float32)

    # Padding is K//2
    P = K // 2
    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_offsets + P - k
            in_bounds = (li >= 0) & (li < L_in)
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=in_bounds & mask_out, other=0.0)
            # Load scalar weight
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            # Accumulate in fp32
            acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    # Store result (cast back to original dtype of y if needed)
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L
    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)  # ReLU
    tl.store(y_ptrs, y_vals, mask=mask_out)


@triton.jit
def split_halves_forward(
    x_full_ptr, x0_ptr, x1_ptr,
    N, C_half, L,
    stride_x_full_n, stride_x_full_c, stride_x_full_l,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    BLOCK_L: tl.constexpr,
):
    # x_full: [N, 2*C_half, L]; write x0[:, :C_half, :] and x1[:, :C_half, :]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    # First half
    x0_ptrs = x_full_ptr + n * stride_x_full_n + ch * stride_x_full_c + l_offsets * stride_x_full_l
    x1_ptrs = x_full_ptr + n * stride_x_full_n + (ch + C_half) * stride_x_full_c + l_offsets * stride_x_full_l
    out0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    out1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l

    x0_vals = tl.load(x0_ptrs, mask=mask_out, other=0.0)
    x1_vals = tl.load(x1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, x0_vals, mask=mask_out)
    tl.store(out1_ptrs, x1_vals, mask=mask_out)


@triton.jit
def concat_halves_forward(
    x0_ptr, x1_ptr, y_full_ptr,
    N, C_half, L,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # y_full: [N, 2*C_half, L]; write x0 into first half and x1 into second half
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    in0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    in1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l
    out0_ptrs = y_full_ptr + n * stride_y_n + ch * stride_y_c + l_offsets * stride_y_l
    out1_ptrs = y_full_ptr + n * stride_y_n + (ch + C_half) * stride_y_c + l_offsets * stride_y_l

    x0_vals = tl.load(in0_ptrs, mask=mask_out, other=0.0)
    x1_vals = tl.load(in1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, x0_vals, mask=mask_out)
    tl.store(out1_ptrs, x1_vals, mask=mask_out)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    BLOCK_L: tl.constexpr,
):
    # mask has shape [N, 1, L], multiply y[i, j, k] *= mask[i, 0, k]
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l
    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(mask_ptrs, mask=mask_out, other=1.0)  # mask is ones (per workload), but read generically
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


def _conv1d_triton(x, w, b, K, BLOCK_L=128):
    """
    Triton wrapper for conv1d forward.
    x: [N, C_in, L_in], w: [C_out, C_in, K], b: [C_out], y: [N, C_out, L_out]
    Returns y. Assumes padding = K//2.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA device."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    N, C_in, L_in = x.shape
    C_out, C_w, K_w = w.shape
    assert C_w == C_in, "Weight in_c must match input channels."
    assert K_w == K, "Kernel size must match."
    P = K // 2
    L_out = L_in + 2 * P

    y = torch.empty((N, C_out, L_out), device=x.device, dtype=torch.float32)  # compute in fp32

    grid = (N, C_out, triton.cdiv(L_out, BLOCK_L))
    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, C_in, C_out, L_in, L_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        K=K, BLOCK_L=BLOCK_L, num_warps=4, num_stages=2
    )
    return y


def _relu_triton(y, BLOCK_L=128):
    N, C, L = y.shape
    y_out = torch.empty_like(y)
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    relu_kernel[grid](
        y, y_out,
        N, C, L,
        y.stride(0), y.stride(1), y.stride(2),
        y_out.stride(0), y_out.stride(1), y_out.stride(2),
        BLOCK_L=BLOCK_L, num_warps=4, num_stages=2
    )
    return y_out


def _split_halves_forward(x_full, C_half, BLOCK_L=128):
    N, C_full, L = x_full.shape
    assert C_full == 2 * C_half, "x_full must be [N, 2*C_half, L]."
    x0 = torch.empty((N, C_half, L), device=x_full.device, dtype=torch.float32)
    x1 = torch.empty((N, C_half, L), device=x_full.device, dtype=torch.float32)
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    split_halves_forward[grid](
        x_full, x0, x1,
        N, C_half, L,
        x_full.stride(0), x_full.stride(1), x_full.stride(2),
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        BLOCK_L=BLOCK_L, num_warps=4, num_stages=2
    )
    return x0, x1


def _concat_halves_forward(x0, x1, C_half, BLOCK_L=128):
    N, C_half, L = x0.shape
    x_full = torch.empty((N, 2 * C_half, L), device=x0.device, dtype=torch.float32)
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    concat_halves_forward[grid](
        x0, x1, x_full,
        N, C_half, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        x_full.stride(0), x_full.stride(1), x_full.stride(2),
        BLOCK_L=BLOCK_L, num_warps=4, num_stages=2
    )
    return x_full


def _mul_mask_triton(y, mask, BLOCK_L=128):
    N, C, L = y.shape
    y_out = torch.empty_like(y)
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    mul_mask_kernel[grid](
        y, mask,
        N, C, L,
        y.stride(0), y.stride(1), y.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_L=BLOCK_L, num_warps=4, num_stages=2
    )
    return y_out


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
    N, C, L = x.shape
    half_channels = C // 2  # 96
    # Ensure x and weights are float32 (compute dtype), mask can be read in any dtype but we'll cast as needed.
    # The conv1d kernels compute in fp32 and write fp32 outputs; we keep the outer tensors as fp32 for simplicity.
    # Note: get_inputs generates weights and biases as float32 by default; we respect that.

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

    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (transforms if not reverse else reversed(transforms)):
        # Split into two halves
        x_full = x  # [N, 2*96, L] where x is [N, 192, L] -> halves
        C_half = half_channels  # 96
        # Split x into x0 and x1
        x0, x1 = _split_halves_forward(x_full, C_half, BLOCK_L=128)

        # conv0: in_c=96, out_c=192, K=5
        h0 = _conv1d_triton(x0, conv0_w, conv0_b, K=5, BLOCK_L=128)
        # ReLU
        h0 = _relu_triton(h0, BLOCK_L=128)

        # conv1: in_c=192, out_c=192, K=5
        h1 = _conv1d_triton(h0, conv1_w, conv1_b, K=5, BLOCK_L=128)
        # ReLU
        h1 = _relu_triton(h1, BLOCK_L=128)

        # conv2: in_c=192, out_c=96, K=5
        h2 = _conv1d_triton(h1, conv2_w, conv2_b, K=5, BLOCK_L=128)  # no ReLU

        # Affine coupling: x1 = x1 + h2
        if not reverse:
            x1 = x1 + h2
        else:
            x1 = x1 - h2

        # Concatenate back
        x_full = _concat_halves_forward(x0, x1, C_half, BLOCK_L=128)

        # Apply mask: y *= x_mask (x_mask is [N, 1, L], broadcast over channels)
        x_full = _mul_mask_triton(x_full, x_mask, BLOCK_L=128)

        # Update original x for next layers (since we are mutating halves)
        x = x_full

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: x, x_mask, reverse, followed by 4 sets of conv weights/bias
        # Mirror the original signature: positional arguments.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
