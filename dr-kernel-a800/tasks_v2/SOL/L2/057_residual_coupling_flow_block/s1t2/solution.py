import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv1d_triton(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, T, K, pad,
    x_sN, x_sC, x_sT,
    w_sO, w_sI, w_sK,
    out_sN, out_sC, out_sT,
    BLOCK_T: tl.constexpr,
):
    """
    Triton implementation of Conv1d (cross-correlation) for stride=1, padding=pad, dilation=1.
    x: [N, IC, T] (float32)
    w: [OC, IC, K] (float32)
    b: [OC] (float32)
    out: [N, OC, T] (float32)
    Grid: (N, OC, ceil_div(T, BLOCK_T))
    Accumulate in float32.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_out_start = pid_block * BLOCK_T
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)
    valid_t = t_out_idx < T

    # Accumulator for output vector of length BLOCK_T
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ic in range(0, IC):
        for k in range(0, K):
            t_in = t_out_idx + k - pad  # valid when 0 <= t_in < T
            valid_t_in = valid_t & (t_in >= 0) & (t_in < T)
            # Load x[n, ic, t_in]
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_t_in, other=0.0)
            # Load w[oc, ic, k]
            w_val = tl.load(w_ptr + pid_oc * w_sO + ic * w_sI + k * w_sK)
            # Multiply and accumulate
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    # Store to out[n, oc, t_out_idx]
    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr + out_offsets, acc, mask=valid_t)


@triton.jit
def multiply_mask_kernel(inp_ptr, mask_ptr, out_ptr,
                          N, C, T,
                          BLOCK: tl.constexpr):
    """
    Elementwise multiply: out[n, c, t] = inp[n, c, t] * mask[n, 0, t].
    Mask is shape [N, 1, T], broadcast over channel dimension.
    Grid: 2D over (N, ceil((C*T)/BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T
    valid = idx < total

    t = idx % T
    c = idx // T

    n = pid_n
    inp_offsets = n * (C * T) + idx
    out_offsets = n * (C * T) + idx

    inp = tl.load(inp_ptr + inp_offsets, mask=valid, other=0.0)
    # mask is [N, T] flat
    mask_offsets = n * T + t
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=valid, other=1.0)

    out = inp * mask_vals
    tl.store(out_ptr + out_offsets, out, mask=valid)


@triton.jit
def relu_kernel(inp_ptr, out_ptr, N, C, T, BLOCK: tl.constexpr):
    """
    Elementwise ReLU: out[n, c, t] = max(inp[n, c, t], 0).
    Grid: 2D over (N, ceil((C*T)/BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T
    valid = idx < total

    t = idx % T
    c = idx // T

    n = pid_n
    offsets = n * (C * T) + idx

    x = tl.load(inp_ptr + offsets, mask=valid, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, x, mask=valid)


@triton.jit
def add_h_kernel(x1_ptr, h_ptr, out_ptr, N, C, T, is_add: tl.constexpr, BLOCK: tl.constexpr):
    """
    Elementwise coupling: out[n, c, t] = x1[n, c, t] + (-+) h[n, c, t].
    is_add: True => out = x1 + h; False => out = x1 - h.
    Grid: 2D over (N, ceil((C*T)/BLOCK))
    """
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)

    start = pid_block * BLOCK
    idx = start + tl.arange(0, BLOCK)
    total = C * T
    valid = idx < total

    t = idx % T
    c = idx // T

    n = pid_n
    offsets = n * (C * T) + idx

    x1 = tl.load(x1_ptr + offsets, mask=valid, other=0.0)
    h = tl.load(h_ptr + offsets, mask=valid, other=0.0)

    if is_add:
        y = x1 + h
    else:
        y = x1 - h

    tl.store(out_ptr + offsets, y, mask=valid)


def _conv1d_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Compute Conv1d (stride=1, padding=pad) using Triton.
    x: [N, IC, T] float32
    w: [OC, IC, K] float32
    b: [OC] float32
    Returns: [N, OC, T] float32
    """
    N, IC, T = x.shape
    OC, IC_w, K = w.shape
    assert IC == IC_w, "Input channels must match weight in_channels"
    assert K == w.shape[2], "Kernel size must be last dim of weight"
    out = torch.empty((N, OC, T), dtype=torch.float32, device=x.device)
    BLOCK_T = 128
    grid = (N, OC, triton.cdiv(T, BLOCK_T))
    conv1d_triton[grid](
        x, w, b, out,
        N, IC, OC, T, K, pad,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T, num_warps=4
    )
    return out


def _multiply_mask_triton(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, T], float32
    mask: [N, 1, T], float32
    Returns: x * mask (broadcast along C)
    """
    assert x.ndim == 3 and mask.ndim == 3
    N, C, T = x.shape
    assert mask.shape == (N, 1, T)
    mask_flat = mask[:, 0, :].contiguous()  # [N, T]
    y = torch.empty_like(x)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    multiply_mask_kernel[grid](x, mask_flat, y, N, C, T, BLOCK=BLOCK, num_warps=4)
    return y


def _relu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Elementwise ReLU on x: [N, C, T]
    """
    N, C, T = x.shape
    y = torch.empty_like(x)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    relu_kernel[grid](x, y, N, C, T, BLOCK=BLOCK, num_warps=4)
    return y


def _add_h_triton(x1: torch.Tensor, h: torch.Tensor, reverse: bool) -> torch.Tensor:
    """
    Elementwise coupling: x1 = x1 + h if not reverse, or x1 - h if reverse.
    """
    assert x1.shape == h.shape and x1.ndim == 3
    N, C, T = x1.shape
    y = torch.empty_like(x1)
    BLOCK = 1024
    grid = (N, triton.cdiv(C * T, BLOCK))
    is_add = not reverse
    add_h_kernel[grid](x1, h, y, N, C, T, is_add=is_add, BLOCK=BLOCK, num_warps=4)
    return y


@triton.jit
def conv1d_write_slice(
    x_ptr, w_ptr, b_ptr, out_ptr_base, oc_offset,
    N, IC, OC, T, K, pad,
    x_sN, x_sC, x_sT,
    w_sO, w_sI, w_sK,
    out_sN, out_sC, out_sT,
    BLOCK_T: tl.constexpr,
):
    """
    Same as conv1d_triton, but writes output to out_ptr_base + oc * out_sC for oc in [oc_offset, oc_offset + OC).
    out_ptr_base: base pointer to out tensor of shape [N, 1, T] (we will slice per oc by offsetting oc in index).
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    oc = pid_oc + oc_offset

    t_out_start = pid_block * BLOCK_T
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)
    valid_t = t_out_idx < T

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    for ic in range(0, IC):
        for k in range(0, K):
            t_in = t_out_idx + k - pad
            valid_t_in = valid_t & (t_in >= 0) & (t_in < T)
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_t_in, other=0.0)
            w_val = tl.load(w_ptr + oc * w_sO + ic * w_sI + k * w_sK)
            acc += x_vals * w_val

    b_val = tl.load(b_ptr + oc)
    acc += b_val

    out_offsets = pid_n * out_sN + oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr_base + out_offsets, acc, mask=valid_t)


def _conv1d_write_slice_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, out_base: torch.Tensor, oc_offset: int, pad: int) -> None:
    """
    Compute Conv1d and write results into out_base[:, oc_offset: oc_offset+OC, :] without creating a new tensor.
    out_base must have shape [N, 1, T] and sufficient channels available at oc_offset+OC.
    """
    N, IC, T = x.shape
    OC, IC_w, K = w.shape
    assert IC == IC_w
    assert out_base.shape[0] == N and out_base.shape[2] == T and out_base.shape[1] == 1
    out = out_base  # we will write into out[:, oc_offset: oc_offset+OC, :]
    BLOCK_T = 128
    grid = (N, OC, triton.cdiv(T, BLOCK_T))
    conv1d_write_slice[grid](
        x, w, b, out, oc_offset,
        N, IC, OC, T, K, pad,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T, num_warps=4
    )


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
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
    transform_3_conv2_weight, transform_3_conv2_bias,
):
    """
    Triton-only implementation of the original forward.
    - All conv1d operations are computed by Triton.
    - Mask application, ReLU, and affine coupling are computed by Triton.
    - No torch.conv1d or torch.cat in the forward.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 for half_channels=96."
    half_channels = C // 2

    # Prepare mask (broadcast along channels)
    mask = x_mask  # [N, 1, T]

    # Final output tensor [N, 192, T], initialized to zeros
    final_out = torch.zeros((N, C, T), dtype=x.dtype, device=x.device)

    # Helper to process one transform and write into final_out
    def process_transform(
        w0, b0, w1, b1, w2, b2,
        is_add_in_coupling: bool
    ):
        # Allocate intermediate tensors; conv results are not stored in global memory.
        # We will compute conv0 on x0 and write to final_out[:, :96, :].
        x0 = x[:, :half_channels, :].contiguous()
        x1 = x[:, half_channels:, :].contiguous()

        # conv0 on x0 -> [N, 96, T]
        conv0 = _conv1d_triton(x0, w0, b0, pad=2)
        conv0 = _multiply_mask_triton(conv0, mask)
        conv0 = _relu_triton(conv0)

        # write conv0 into final_out[:, :96, :]
        _conv1d_write_slice_triton(x0, w0, b0, final_out, 0, pad=2)

        # conv1 on conv0 -> [N, 192, T]
        conv1_in = conv0
        conv1 = _conv1d_triton(conv1_in, w1, b1, pad=2)
        conv1 = _multiply_mask_triton(conv1, mask)
        conv1 = _relu_triton(conv1)

        # write conv1 into final_out[:, 96:, :]
        _conv1d_write_slice_triton(conv1_in, w1, b1, final_out, half_channels, pad=2)

        # conv2 on conv1_out -> [N, 96, T] (note: conv2 input is conv1 output, which is ReLUed)
        conv2_in = conv1  # but for conv2 we need the ReLUed conv1, we already have it from above
        conv2 = _conv1d_triton(conv2_in, w2, b2, pad=2)
        conv2 = _multiply_mask_triton(conv2, mask)
        conv2 = _relu_triton(conv2)  # h for coupling

        # coupling: update x1 slice
        # x1 is final_out[:, half_channels:, :]
        x1_slice = final_out[:, half_channels:, :].contiguous()
        h_slice = conv2  # shape [N, 96, T]
        if is_add_in_coupling:
            new_x1 = _add_h_triton(x1_slice, h_slice, reverse=False)
        else:
            new_x1 = _add_h_triton(x1_slice, h_slice, reverse=True)
        final_out[:, half_channels:, :] = new_x1

    # Apply transforms in forward order (reverse=False), and in reverse order if reverse=True
    if not reverse:
        # transform 0
        process_transform(transform_0_conv0_weight, transform_0_conv0_bias,
                          transform_0_conv1_weight, transform_0_conv1_bias,
                          transform_0_conv2_weight, transform_0_conv2_bias,
                          is_add_in_coupling=True)
        # transform 1
        process_transform(transform_1_conv0_weight, transform_1_conv0_bias,
                          transform_1_conv1_weight, transform_1_conv1_bias,
                          transform_1_conv2_weight, transform_1_conv2_bias,
                          is_add_in_coupling=True)
        # transform 2
        process_transform(transform_2_conv0_weight, transform_2_conv0_bias,
                          transform_2_conv1_weight, transform_2_conv1_bias,
                          transform_2_conv2_weight, transform_2_conv2_bias,
                          is_add_in_coupling=True)
        # transform 3
        process_transform(transform_3_conv0_weight, transform_3_conv0_bias,
                          transform_3_conv1_weight, transform_3_conv1_bias,
                          transform_3_conv2_weight, transform_3_conv2_bias,
                          is_add_in_coupling=True)
    else:
        # transform 3 first (reverse order)
        process_transform(transform_3_conv0_weight, transform_3_conv0_bias,
                          transform_3_conv1_weight, transform_3_conv1_bias,
                          transform_3_conv2_weight, transform_3_conv2_bias,
                          is_add_in_coupling=False)
        # transform 2
        process_transform(transform_2_conv0_weight, transform_2_conv0_bias,
                          transform_2_conv1_weight, transform_2_conv1_bias,
                          transform_2_conv2_weight, transform_2_conv2_bias,
                          is_add_in_coupling=False)
        # transform 1
        process_transform(transform_1_conv0_weight, transform_1_conv0_bias,
                          transform_1_conv1_weight, transform_1_conv1_bias,
                          transform_1_conv2_weight, transform_1_conv2_bias,
                          is_add_in_coupling=False)
        # transform 0
        process_transform(transform_0_conv0_weight, transform_0_conv0_bias,
                          transform_0_conv1_weight, transform_0_conv1_bias,
                          transform_0_conv2_weight, transform_0_conv2_bias,
                          is_add_in_coupling=False)

    return final_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same signature as original run:
        # x, x_mask, reverse, then 24 weight/bias tensors in order.
        # This function simply calls run, ensuring Triton-only computation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
