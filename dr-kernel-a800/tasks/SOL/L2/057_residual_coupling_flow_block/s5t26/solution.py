import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_bias_relu(
    x_ptr,         # *f32, [B, Cin, T]
    w_ptr,         # *f32, [Cout, Cin*K]
    b_ptr,         # *f32, [Cout]
    y_ptr,         # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,  # padding = (K-1)//2, here 2 for K=5
    BLOCK_T: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_t_block = tl.program_id(2)  # time block

    # compute time indices for this block
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # accumulate output
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel elements
    # for each c_in, add contributions for k=0..K-1
    # y[b, co, t] = sum_{c_in, k} x[b, c_in, t - P + k] * w[co, c_in*K + k]
    for c_in in range(Cin):
        for k in range(K):
            t_in = t_offsets - P + k
            valid = (t_in >= 0) & (t_in < T) & mask_t
            x_index = (((pid_b * Cin) + c_in) * T) + t_in
            x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
            w_index = pid_co * (Cin * K) + c_in * K + k
            w_val = tl.load(w_ptr + w_index)
            acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store to y[b, co, t]
    y_index = (((pid_b * Cout) + pid_co) * T) + t_offsets
    tl.store(y_ptr + y_index, acc, mask=mask_t)


@triton.jit
def add_mask_to_h_triton(
    h_ptr,       # *f32, [B, C1, T]
    mask_ptr,    # *f32, [B, 1, T]
    out_ptr,     # *f32, [B, C1, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    h_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    # mask has shape [B, 1, T], so we load mask[pid_b, 0, t]
    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    out_val = h_val * mask_val
    tl.store(out_ptr + h_index, out_val, mask=mask_t)


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,     # *f32, [B, C1, T]
    h_ptr,      # *f32, [B, C1, T]
    out_ptr,    # *f32, [B, C1, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    ADD: tl.constexpr,  # True means out = x1 + h; False means out = x1 - h
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    h_index = x1_index  # same layout

    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    out_val = x1_val + h_val if ADD else x1_val - h_val
    tl.store(out_ptr + x1_index, out_val, mask=mask_t)


@triton.jit
def concat_copy_first_half(
    x0_ptr,       # *f32, [B, C0, T]
    out_ptr,      # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # copy from x0[b, c, t] to out[b, c, t]
    x0_index = (((pid_b * C0) + pid_c) * T) + t_offsets
    out_index = (((pid_b * C0) + pid_c) * T) + t_offsets
    x0_val = tl.load(x0_ptr + x0_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, x0_val, mask=mask_t)


@triton.jit
def concat_copy_second_half(
    x1_ptr,       # *f32, [B, C1, T]
    out_ptr,      # *f32, [B, C, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    C0: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # copy from x1[b, c, t] to out[b, C0 + c, t]
    x1_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    out_index = (((pid_b * (C0 + C1)) + (pid_c + C0)) * T) + t_offsets
    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, x1_val, mask=mask_t)


@triton.jit
def apply_mask_to_out_triton(
    out_ptr,      # *f32, [B, C, T]
    mask_ptr,     # *f32, [B, 1, T]
    out_ptr_out,  # *f32, [B, C, T]
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    out_index = (((pid_b * C) + pid_c) * T) + t_offsets
    out_val = tl.load(out_ptr + out_index, mask=mask_t, other=0.0)

    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    out_val = out_val * mask_val
    tl.store(out_ptr_out + out_index, out_val, mask=mask_t)


def _conv1d_triton_stride1_bias_relu(x, w, b):
    """
    Compute y = ReLU(conv1d(x, w, padding=(K-1)//2, stride=1, bias=b)) using Triton.
    x: [B, Cin, T], w: [Cout, Cin*K], b: [Cout], all float32 on CUDA.
    Returns y: [B, Cout, T].
    """
    B, Cin, T = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w, "Incompatible weights for conv1d"
    P = (K - 1) // 2
    y = torch.empty((B, Cout, T), device=x.device, dtype=torch.float32)
    BLOCK_T = 128
    grid = (B, Cout, triton.cdiv(T, BLOCK_T))
    conv1d_stride1_bias_relu[grid](
        x, w, b, y,
        B, Cin, Cout, T, K, P, BLOCK_T,
        num_warps=4, num_stages=2
    )
    return y


def _apply_mask_to_h_triton(h, x_mask):
    """
    h: [B, C1, T], x_mask: [B, 1, T] float32 CUDA tensors.
    Returns h_masked: [B, C1, T]
    """
    B, C1, T = h.shape
    h_masked = torch.empty_like(h)
    BLOCK_T = 128
    grid = (B, C1, triton.cdiv(T, BLOCK_T))
    add_mask_to_h_triton[grid](
        h, x_mask, h_masked,
        B, C1, T, BLOCK_T,
        num_warps=4, num_stages=2
    )
    return h_masked


def _add_h_to_x1_triton(x1, h_masked, add: bool):
    """
    Update x1: out = x1 + h if add=True else out = x1 - h
    x1, h_masked: [B, C1, T] float32 CUDA tensors
    Returns out: [B, C1, T]
    """
    B, C1, T = x1.shape
    out = torch.empty_like(x1)
    BLOCK_T = 128
    grid = (B, C1, triton.cdiv(T, BLOCK_T))
    add_h_to_x1_triton[grid](
        x1, h_masked, out,
        B, C1, T, add, BLOCK_T,
        num_warps=4, num_stages=2
    )
    return out


def _concat_copy_first_half(x0, out, C0):
    """
    Copy x0 [B, C0, T] into out [B, C0, T]. out must have shape [B, C0, T].
    """
    B, C0, T = x0.shape
    # We assume out already allocated as [B, C0, T]
    BLOCK_T = 128
    grid = (B, C0, triton.cdiv(T, BLOCK_T))
    concat_copy_first_half[grid](
        x0, out,
        B, C0, T, BLOCK_T,
        num_warps=4, num_stages=2
    )


def _concat_copy_second_half(x1, out, C1, C0):
    """
    Copy x1 [B, C1, T] into out [B, C1, T] but at columns [C0:C0+C1) of out.
    out must have shape [B, C0 + C1, T], x1 has shape [B, C1, T].
    """
    B, C1, T = x1.shape
    # out shape is [B, C0 + C1, T]
    # grid over (B, C1, blocks over T)
    BLOCK_T = 128
    grid = (B, C1, triton.cdiv(T, BLOCK_T))
    concat_copy_second_half[grid](
        x1, out,
        B, C1, T, C0, BLOCK_T,
        num_warps=4, num_stages=2
    )


def _apply_mask_to_out_triton(out, x_mask):
    """
    out: [B, C, T], x_mask: [B, 1, T] float32 CUDA tensors.
    Returns out_masked: [B, C, T] = out * x_mask (broadcast across channels).
    """
    B, C, T = out.shape
    out_masked = torch.empty_like(out)
    BLOCK_T = 128
    grid = (B, C, triton.cdiv(T, BLOCK_T))
    apply_mask_to_out_triton[grid](
        out, x_mask, out_masked,
        B, C, T, BLOCK_T,
        num_warps=4, num_stages=2
    )
    return out_masked


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
    Triton-only implementation of the residual coupling flow block.
    x: [B, 2*half_channels, T]
    x_mask: [B, 1, T]
    forward: x1 = x1 + transform(x0)
    reverse: x1 = x1 - transform(x0) (in reverse order)
    """
    # Ensure CUDA tensors and float32
    device = x.device
    dtype = torch.float32
    if x.dtype != dtype:
        x = x.to(dtype)
    if x_mask.dtype != dtype:
        x_mask = x_mask.to(dtype)
    B, C, T = x.shape
    half_channels = C // 2

    # Launch Triton kernels: 4 transforms, each with 3 convs (conv0->ReLU->conv1->ReLU->conv2)
    # We will perform the forward updates and concatenation with Triton. In reverse, we subtract.

    if not reverse:
        # Forward pass: apply transformations sequentially
        # transform 0
        x0 = x[:, :half_channels, :].contiguous()
        x1 = x[:, half_channels:, :].contiguous()
        # conv0 -> ReLU -> conv1 -> ReLU -> conv2
        y0 = _conv1d_triton_stride1_bias_relu(x0, transform_0_conv0_weight, transform_0_conv0_bias)
        y1 = _conv1d_triton_stride1_bias_relu(y0, transform_0_conv1_weight, transform_0_conv1_bias)
        y2 = _conv1d_triton_stride1_bias_relu(y1, transform_0_conv2_weight, transform_0_conv2_bias)
        # mask
        h_masked = _apply_mask_to_h_triton(y2, x_mask)
        # update x1
        x1 = _add_h_to_x1_triton(x1, h_masked, add=True)
        # concatenate
        out = torch.empty((B, C, T), device=device, dtype=dtype)
        _concat_copy_first_half(x0, out, half_channels)
        _concat_copy_second_half(x1, out, half_channels, half_channels)
        # apply x_mask across channels (broadcast along time)
        out = _apply_mask_to_out_triton(out, x_mask)
        # save out for subsequent transforms
        x = out
    else:
        # Reverse pass: apply transforms in reverse order (subtract h)
        # We keep x as the original input [B, 2*half, T] and update it in place after each transform.
        # transform 3
        x0 = x[:, :half_channels, :].contiguous()
        x1 = x[:, half_channels:, :].contiguous()
        y0 = _conv1d_triton_stride1_bias_relu(x0, transform_3_conv0_weight, transform_3_conv0_bias)
        y1 = _conv1d_triton_stride1_bias_relu(y0, transform_3_conv1_weight, transform_3_conv1_bias)
        y2 = _conv1d_triton_stride1_bias_relu(y1, transform_3_conv2_weight, transform_3_conv2_bias)
        h_masked = _apply_mask_to_h_triton(y2, x_mask)
        x1 = _add_h_to_x1_triton(x1, h_masked, add=False)  # subtract
        out = torch.empty((B, C, T), device=device, dtype=dtype)
        _concat_copy_first_half(x0, out, half_channels)
        _concat_copy_second_half(x1, out, half_channels, half_channels)
        out = _apply_mask_to_out_triton(out, x_mask)
        x = out

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: x, x_mask, reverse, then 24 weight/bias tensors
        # We need to forward using Triton kernels for all computation.
        # Construct the argument list; the caller of ModelNew will pass the same inputs as the original run.
        # Ensure inputs are on CUDA and float32 for Triton
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False

        # We assume all weights and biases are provided and already on the same device as x
        # ModelNew.forward must call Triton kernels. We do not use any torch ops for computation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
