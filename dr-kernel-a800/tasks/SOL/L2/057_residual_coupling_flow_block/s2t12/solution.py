import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K, P,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # program ids: over (N, C_out, tiles of L_out)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    # output time offsets
    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # initialize accumulator with bias[co]
    b_val = tl.load(b_ptr + co)
    acc = tl.full((BLOCK_L,), 0.0, tl.float32) + b_val  # cast bias to fp32 accumulator

    # loop over input channels and kernel taps
    for ci in range(0, C_in):
        # for each kernel tap k
        for k in range(0, K):
            # compute input indices li = lo + P - k
            li = l_out_offsets + P - k
            in_bounds = (li >= 0) & (li < L_in) & mask_out

            # load x[n, ci, li] if in_bounds
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

            # load w[co, ci, k]
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_vals = tl.load(w_ptrs)  # scalar weight

            # accumulate
            acc += x_vals * w_vals

    # store result
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


@triton.jit
def relu_triton_kernel(
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
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask_out)


@triton.jit
def concat_halves_backward(
    y2c_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    # y2c: [N, 2*C_half, L]; split along channel dim
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y0_ptrs = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y1_ptrs = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l
    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(y0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(y1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y2c_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    BLOCK_L: tl.constexpr,
):
    # y0: [N, C_half, L], y1: [N, C_half, L]; write y2c: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    y2c_ptrs0 = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y2c_ptrs1 = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l

    y0_vals = tl.load(out0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(out1_ptrs, mask=mask_out, other=0.0)
    tl.store(y2c_ptrs0, y0_vals, mask=mask_out)
    tl.store(y2c_ptrs1, y1_vals, mask=mask_out)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,  # mask shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_m_n + 0 * stride_m_c + l_offsets * stride_m_l
    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_out, other=1.0)
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


def single_transform_triton(
    x_curr,                 # [N, C, L] current x (to split into halves)
    mask,                   # [N, 1, L] x_mask
    w0, b0,                 # conv0 weights/bias (C_in=96, C_out=192, K=5)
    w1, b1,                 # conv1 weights/bias (C_in=192, C_out=192, K=5)
    w2, b2,                 # conv2 weights/bias (C_in=192, C_out=96, K=5)
    reverse: bool,          # whether to subtract h
    device,                 # Triton requires CUDA tensors
):
    # Ensure contiguous for predictable strides
    x_curr = x_curr.contiguous()
    N, C, L = x_curr.shape
    half = C // 2

    # Split into x0 and x1
    x0 = torch.empty((N, half, L), device=device, dtype=x_curr.dtype)
    x1 = torch.empty((N, half, L), device=device, dtype=x_curr.dtype)

    grid_split = (N, half, triton.cdiv(L, 128))
    concat_halves_backward[grid_split](
        x_curr, x0, x1,
        N, half, L,
        x_curr.stride(0), x_curr.stride(1), x_curr.stride(2),
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Conv0: h0 = conv1d(x0, w0, b0) + ReLU
    h0 = torch.empty((N, w0.shape[0], L), device=device, dtype=x_curr.dtype)  # C_out = 192
    grid_c0 = (N, w0.shape[0], triton.cdiv(L, 128))
    conv1d_kernel[grid_c0](
        x0, w0, b0, h0,
        N, w0.shape[1], w0.shape[0], x0.shape[2], h0.shape[2], w0.shape[2], w0.shape[2] // 2,
        x0.stride(0), x0.stride(1), x0.stride(2),
        w0.stride(0), w0.stride(1), w0.stride(2),
        h0.stride(0), h0.stride(1), h0.stride(2),
        BLOCK_L=128, num_warps=4
    )
    # ReLU
    relu_triton_kernel[grid_c0](
        h0, h0,
        N, w0.shape[0], L,
        h0.stride(0), h0.stride(1), h0.stride(2),
        h0.stride(0), h0.stride(1), h0.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Conv1: h1 = conv1d(h0, w1, b1) + ReLU
    h1 = torch.empty((N, w1.shape[0], L), device=device, dtype=x_curr.dtype)  # C_out = 192
    grid_c1 = (N, w1.shape[0], triton.cdiv(L, 128))
    conv1d_kernel[grid_c1](
        h0, w1, b1, h1,
        N, w1.shape[1], w1.shape[0], h0.shape[2], h1.shape[2], w1.shape[2], w1.shape[2] // 2,
        h0.stride(0), h0.stride(1), h0.stride(2),
        w1.stride(0), w1.stride(1), w1.stride(2),
        h1.stride(0), h1.stride(1), h1.stride(2),
        BLOCK_L=128, num_warps=4
    )
    # ReLU
    relu_triton_kernel[grid_c1](
        h1, h1,
        N, w1.shape[0], L,
        h1.stride(0), h1.stride(1), h1.stride(2),
        h1.stride(0), h1.stride(1), h1.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Conv2: h2 = conv1d(h1, w2, b2) (no ReLU)
    h2 = torch.empty((N, w2.shape[0], L), device=device, dtype=x_curr.dtype)  # C_out = 96
    grid_c2 = (N, w2.shape[0], triton.cdiv(L, 128))
    conv1d_kernel[grid_c2](
        h1, w2, b2, h2,
        N, w2.shape[1], w2.shape[0], h1.shape[2], h2.shape[2], w2.shape[2], w2.shape[2] // 2,
        h1.stride(0), h1.stride(1), h1.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        h2.stride(0), h2.stride(1), h2.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # h2 shape: [N, 96, L], x1 shape: [N, 96, L]
    # Forward: x1 = x1 + h2; Reverse: x1 = x1 - h2
    if reverse:
        x1 = x1 - h2
    else:
        x1 = x1 + h2

    # Concatenate [x0, x1] back
    x_out = torch.empty((N, 192, L), device=device, dtype=x_curr.dtype)
    grid_concat = (N, 96, triton.cdiv(L, 128))
    concat_halves_forward[grid_concat](
        x0, x1, x_out,
        N, 96, L,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        x_out.stride(0), x_out.stride(1), x_out.stride(2),
        BLOCK_L=128, num_warps=4
    )

    # Apply mask: x_out *= mask
    grid_mask = (N, 192, triton.cdiv(L, 128))
    mul_mask_kernel[grid_mask](
        x_out, mask,
        N, 192, L,
        x_out.stride(0), x_out.stride(1), x_out.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_L=128, num_warps=4
    )

    return x_out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_mask, reverse, *args):
        # args contains all weights and biases for 4 transforms
        # unpack them into transforms list
        # We have 4 transforms, each with 3 convs: (w0, b0), (w1, b1), (w2, b2)
        # Number of args per transform is 6
        num_transforms = 4
        transforms = []
        for i in range(num_transforms):
            # indices for this transform
            start = i * 6
            w0 = args[start]
            b0 = args[start + 1]
            w1 = args[start + 2]
            b1 = args[start + 3]
            w2 = args[start + 4]
            b2 = args[start + 5]
            transforms.append((w0, b0, w1, b1, w2, b2))

        # Current x
        x_curr = x

        for i, (w0, b0, w1, b1, w2, b2) in enumerate(transforms):
            x_curr = single_transform_triton(
                x_curr, x_mask, w0, b0, w1, b1, w2, b2, reverse, x_curr.device
            )

        return x_curr


def run(*args):
    return ModelNew()(*args)
