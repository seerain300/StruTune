import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # program ids
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    # output time offsets
    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # initialize accumulator with bias[co] (scalar)
    b_val = tl.load(b_ptr + co)  # float32 bias
    acc = tl.full((BLOCK_L,), b_val, dtype=tl.float32)

    # loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            # li = lo + P - k; with padding P = K//2 (default)
            li = l_out_offsets + (K // 2) - k
            # in-bounds mask for x
            mask_in = (li >= 0) & (li < L_in) & mask_out
            # load x[n, ci, li] with mask
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=mask_in, other=0.0)  # load as original dtype, masked
            x_vals_f32 = x_vals.to(tl.float32)

            # load weight scalar w[co, ci, k]
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar, typically float32
            w_val_f32 = w_val.to(tl.float32)

            # accumulate
            acc += x_vals_f32 * w_val_f32

    # store results y[n, co, l_out_offsets]
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
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
    mask = l_offsets < L
    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def split_halves_forward(
    x_full_ptr, x0_ptr, x1_ptr,
    N, C_half, L,
    stride_full_n, stride_full_c, stride_full_l,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    BLOCK_L: tl.constexpr,
):
    # x_full: [N, 2*C_half, L]; write x0[:, :C_half, :], x1[:, :C_half, :]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    # first half
    full0_ptrs = x_full_ptr + n * stride_full_n + ch * stride_full_c + l_offsets * stride_full_l
    out0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    x0_vals = tl.load(full0_ptrs, mask=mask, other=0.0)
    tl.store(out0_ptrs, x0_vals, mask=mask)

    # second half
    full1_ptrs = x_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + l_offsets * stride_full_l
    out1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l
    x1_vals = tl.load(full1_ptrs, mask=mask, other=0.0)
    tl.store(out1_ptrs, x1_vals, mask=mask)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_full_n, stride_full_c, stride_full_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    out0_ptrs = y_full_ptr + n * stride_full_n + ch * stride_full_c + l_offsets * stride_full_l
    in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + l_offsets * stride_full_l
    in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(in0_ptrs, mask=mask, other=0.0)
    y1_vals = tl.load(in1_ptrs, mask=mask, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask)
    tl.store(out1_ptrs, y1_vals, mask=mask)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr, out_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l
    y_vals = tl.load(y_ptrs, mask=mask_l, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_l, other=1.0)
    out_vals = y_vals * m_vals
    out_ptrs = out_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l  # out == y
    tl.store(out_ptrs, out_vals, mask=mask_l)


@triton.jit
def conv1d_forward_kernel_v2(  # alternative if needed (not used in this run)
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # This is a copy of conv1d_forward_kernel; we keep it in case we need to vary parameters.
    pass


def _conv1d_triton(x, w, b, padding=2, stride=1):
    """
    x: [N, C_in, L_in], w: [C_out, C_in, K], b: [C_out]
    Returns y: [N, C_out, L_out], with L_out = floor((L_in + 2*padding - K)/stride) + 1
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    N, C_in, L_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, "Weight input channels must match x channels"
    # Output length
    P = padding
    L_out = (L_in + 2 * P - K) // stride + 1
    y = torch.empty((N, C_out, L_out), device=x.device, dtype=torch.float32)

    # Strides
    stride_x_n, stride_x_c, stride_x_l = x.stride()
    stride_w_co, stride_w_ci, stride_w_k = w.stride()
    stride_y_n, stride_y_c, stride_y_l = y.stride()

    # grid over (N, C_out, tiles along L_out)
    BLOCK_L = 128
    grid = (N, C_out, triton.cdiv(L_out, BLOCK_L))

    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, C_in, C_out, L_in, L_out,
        stride_x_n, stride_x_c, stride_x_l,
        stride_w_co, stride_w_ci, stride_w_k,
        stride_y_n, stride_y_c, stride_y_l,
        K=K, BLOCK_L=BLOCK_L, num_warps=4,
    )
    return y


def _relu_triton(y):
    y = y.contiguous()
    N, C, L = y.shape
    y_out = torch.empty_like(y)
    stride_y_n, stride_y_c, stride_y_l = y.stride()
    stride_out_n, stride_out_c, stride_out_l = y_out.stride()
    BLOCK_L = 128
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    relu_kernel[grid](y, y_out, N, C, L, stride_y_n, stride_y_c, stride_y_l, stride_out_n, stride_out_c, stride_out_l, BLOCK_L=BLOCK_L, num_warps=4)
    return y_out


def _split_triton(x_full, C_half):
    """
    x_full: [N, 2*C_half, L]
    Returns x0: [N, C_half, L], x1: [N, C_half, L]
    """
    x_full = x_full.contiguous()
    N, C2, L = x_full.shape
    assert C2 == 2 * C_half
    x0 = torch.empty((N, C_half, L), device=x_full.device, dtype=x_full.dtype)
    x1 = torch.empty((N, C_half, L), device=x_full.device, dtype=x_full.dtype)

    stride_full_n, stride_full_c, stride_full_l = x_full.stride()
    stride_x0_n, stride_x0_c, stride_x0_l = x0.stride()
    stride_x1_n, stride_x1_c, stride_x1_l = x1.stride()
    BLOCK_L = 128
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    split_halves_forward[grid](
        x_full, x0, x1,
        N, C_half, L,
        stride_full_n, stride_full_c, stride_full_l,
        stride_x0_n, stride_x0_c, stride_x0_l,
        stride_x1_n, stride_x1_c, stride_x1_l,
        BLOCK_L=BLOCK_L, num_warps=4,
    )
    return x0, x1


def _concat_triton(y0, y1):
    """
    y0: [N, C_half, L], y1: [N, C_half, L]
    Returns y_full: [N, 2*C_half, L]
    """
    y0 = y0.contiguous()
    y1 = y1.contiguous()
    N, C_half, L = y0.shape
    y_full = torch.empty((N, 2 * C_half, L), device=y0.device, dtype=y0.dtype)
    stride_y0_n, stride_y0_c, stride_y0_l = y0.stride()
    stride_y1_n, stride_y1_c, stride_y1_l = y1.stride()
    stride_full_n, stride_full_c, stride_full_l = y_full.stride()
    BLOCK_L = 128
    grid = (N, C_half, triton.cdiv(L, BLOCK_L))
    concat_halves_forward[grid](
        y0, y1, y_full,
        N, C_half, L,
        stride_y0_n, stride_y0_c, stride_y0_l,
        stride_y1_n, stride_y1_c, stride_y1_l,
        stride_full_n, stride_full_c, stride_full_l,
        BLOCK_L=BLOCK_L, num_warps=4,
    )
    return y_full


def _mul_mask_triton(y, mask):
    """
    y: [N, C, L], mask: [N, 1, L]
    Returns y * mask
    """
    y = y.contiguous()
    mask = mask.contiguous()
    N, C, L = y.shape
    out = torch.empty_like(y)
    stride_y_n, stride_y_c, stride_y_l = y.stride()
    stride_mask_n, stride_mask_c, stride_mask_l = mask.stride()
    BLOCK_L = 128
    grid = (N, C, triton.cdiv(L, BLOCK_L))
    mul_mask_kernel[grid](
        y, mask, out,
        N, C, L,
        stride_y_n, stride_y_c, stride_y_l,
        stride_mask_n, stride_mask_c, stride_mask_l,
        BLOCK_L=BLOCK_L, num_warps=4,
    )
    return out


@torch.no_grad()
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
    Triton-optimized forward/reverse of the original 'run' function.
    All computations done by Triton kernels: Conv1d (forward), ReLU, split/concat, mask multiplication.
    """
    # Ensure device is CUDA and dtype float32 for numerical stability
    assert x.is_cuda, "Input x must be on CUDA device"
    # We'll operate in float32
    if x.dtype != torch.float32:
        x = x.float()
    # half_channels
    C = x.shape[1]
    half_channels = C // 2

    # Prepare list of transforms
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

    # Full concatenated tensor: [N, 2*half_channels, L]
    # Initially, set x_full to x
    x_full = x.clone()  # [N, C, L], C=192

    # Process transforms
    if not reverse:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split halves
            x0, x1 = _split_triton(x_full, half_channels)

            # Conv0: y0 = conv1d(x0, conv0_w, conv0_b), padding=2, stride=1
            y0 = _conv1d_triton(x0, conv0_w, conv0_b, padding=2, stride=1)
            # ReLU
            y0 = _relu_triton(y0)

            # Conv1: y1 = conv1d(y0, conv1_w, conv1_b)
            y1 = _conv1d_triton(y0, conv1_w, conv1_b, padding=2, stride=1)
            # ReLU
            y1 = _relu_triton(y1)

            # Conv2: y2 = conv1d(y1, conv2_w, conv2_b)
            y2 = _conv1d_triton(y1, conv2_w, conv2_b, padding=2, stride=1)

            # Affine coupling: x1 = x1 + y2 (no ReLU after conv2)
            # Ensure x1 and y2 are same dtype
            x1 = x1 + y2

            # Concatenate back
            x_full = _concat_triton(x0, x1)

            # Apply mask (mask is [N,1,L], provided as x_mask; in typical setup ones)
            # If mask not ones, multiply
            x_full = _mul_mask_triton(x_full, x_mask)

    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split halves
            x0, x1 = _split_triton(x_full, half_channels)

            # Conv0: y0 = conv1d(x0, conv0_w, conv0_b), padding=2, stride=1
            y0 = _conv1d_triton(x0, conv0_w, conv0_b, padding=2, stride=1)
            # ReLU
            y0 = _relu_triton(y0)

            # Conv1: y1 = conv1d(y0, conv1_w, conv1_b)
            y1 = _conv1d_triton(y0, conv1_w, conv1_b, padding=2, stride=1)
            # ReLU
            y1 = _relu_triton(y1)

            # Conv2: y2 = conv1d(y1, conv2_w, conv2_b)
            y2 = _conv1d_triton(y1, conv2_w, conv2_b, padding=2, stride=1)

            # Inverse affine coupling: x1 = x1 - y2
            x1 = x1 - y2

            # Concatenate back
            x_full = _concat_triton(x0, x1)

            # Apply mask (mask is [N,1,L])
            x_full = _mul_mask_triton(x_full, x_mask)

    return x_full


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args expected: x, x_mask, reverse, then 24 weight/bias tensors for 4 transforms.
        # We will pass all args to the Triton-optimized run function.
        return run_triton(*args)

# Example usage:
# model = ModelNew().cuda()
# inputs = get_inputs({'batch_size': 8, 'time': 768}, torch.device('cuda'))
# x = inputs['x']
# x_mask = inputs['x_mask']
# reverse = False
# # pass weights and biases
# # model(*args) where args are as above


def run(*args):
    return ModelNew()(*args)
