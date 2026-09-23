import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _pick_block_t(T):
    # choose a reasonable block size for time dimension
    if T >= 4096:
        return 256
    elif T >= 2048:
        return 128
    elif T >= 1024:
        return 128
    else:
        return 64


@triton.jit
def conv1d_triton_stride1_bias_relu(
    x_ptr,        # *f32, [B, Cin, T]
    w_ptr,        # *f32, [Cout, Cin*K]
    b_ptr,        # *f32, [Cout]
    y_ptr,        # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,           # kernel size
    PAD: tl.constexpr,         # padding = (K - 1) // 2
    BLOCK_T: tl.constexpr,     # block size for time dimension
):
    # program ids
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t_block = tl.program_id(2)

    # time offsets for this block
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # initialize output accumulator
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel taps
    for c_in in range(0, Cin):
        # for each k in [0, K)
        for k in tl.static_range(0, K):
            t_in = t_offsets - PAD + k
            in_range = (t_in >= 0) & (t_in < T) & mask_t
            # load x[b, c_in, t_in]
            x_index = (pid_b * Cin * T) + c_in * T + t_in
            x_val = tl.load(x_ptr + x_index, mask=in_range, other=0.0)
            # load corresponding weight w[co, c_in*K + k]
            w_index = pid_co * (Cin * K) + c_in * K + k
            w_val = tl.load(w_ptr + w_index)
            acc += x_val * w_val

    # add bias and apply ReLU
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val
    acc = tl.maximum(acc, 0.0)  # ReLU

    # store to y[b, co, t_offsets]
    y_index = (pid_b * Cout * T) + pid_co * T + t_offsets
    tl.store(y_ptr + y_index, acc, mask=mask_t)


@triton.jit
def apply_mask_to_h(
    h_ptr,        # *f32, [B, C1, T]
    mask_ptr,     # *f32, [B, 1, T]
    h_out_ptr,    # *f32, [B, C1, T]
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

    h_index = ((pid_b * C1) + pid_c) * T + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    h_val = h_val * mask_val
    tl.store(h_out_ptr + h_index, h_val, mask=mask_t)


@triton.jit
def add_h_to_x1(
    x1_ptr,       # *f32, [B, C1, T]
    h_ptr,        # *f32, [B, C1, T]
    x1_out_ptr,   # *f32, [B, C1, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    ADD: tl.constexpr,  # True: x1 = x1 + h, False: x1 = x1 - h
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = ((pid_b * C1) + pid_c) * T + t_offsets
    h_index = ((pid_b * C1) + pid_c) * T + t_offsets

    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    if ADD:
        out_val = x1_val + h_val
    else:
        out_val = x1_val - h_val

    tl.store(x1_out_ptr + x1_index, out_val, mask=mask_t)


@triton.jit
def concat_copy_first_half(
    x0_ptr,       # *f32, [B, C0, T]
    out_ptr,      # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # 0..C0-1
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x0_index = (pid_b * C0 * T) + pid_c * T + t_offsets
    out_index = (pid_b * (C0 + C1) * T) + pid_c * T + t_offsets

    val = tl.load(x0_ptr + x0_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def concat_copy_second_half(
    x1_ptr,       # *f32, [B, C1, T]
    out_ptr,      # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # 0..C1-1
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (pid_b * C1 * T) + pid_c * T + t_offsets
    out_index = (pid_b * (C0 + C1) * T) + (pid_c + C0) * T + t_offsets

    val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


def _conv1d_triton_stride1_bias_relu(x, w, b):
    """
    Triton conv1d with stride=1, padding=(K-1)//2, bias, ReLU.
    x: [B, Cin, T], w: [Cout, Cin*K], b: [Cout], float32 CUDA tensors.
    Returns y: [B, Cout, T].
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be CUDA"
    B, Cin, T = x.shape
    Cout, Cin_w, K = w.shape
    assert Cin == Cin_w, "Incompatible weights for conv"
    P = (K - 1) // 2  # padding
    y = torch.empty((B, Cout, T), dtype=torch.float32, device=x.device)

    BLOCK_T = _pick_block_t(T)
    grid = (B, Cout, triton.cdiv(T, BLOCK_T))
    conv1d_triton_stride1_bias_relu[grid](
        x, w, b, y,
        B=B, Cin=Cin, Cout=Cout, T=T,
        K=K, PAD=P, BLOCK_T=BLOCK_T,
    )
    return y


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

    out_index = (pid_b * C * T) + pid_c * T + t_offsets
    out_val = tl.load(out_ptr + out_index, mask=mask_t, other=0.0)

    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    out_val = out_val * mask_val
    tl.store(out_ptr_out + out_index, out_val, mask=mask_t)


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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    Triton-only implementation: all convs, masks, and concatenations are Triton kernels.
    """
    assert x.is_cuda, "Input x must be on CUDA for Triton kernels"
    B, C, T = x.shape
    half_channels = C // 2
    assert C == 192 and half_channels == 96, "Expected C=192, half_channels=96"

    # Helpers to apply one transform using Triton conv1d (stride=1, padding=(K-1)//2), ReLU fused
    def apply_one_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, ADD: bool):
        # conv0
        h = _conv1d_triton_stride1_bias_relu(x0, conv0_w, conv0_b)  # [B, hidden_channels, T]
        # conv1
        h = _conv1d_triton_stride1_bias_relu(h, conv1_w, conv1_b)
        # conv2
        h = _conv1d_triton_stride1_bias_relu(h, conv2_w, conv2_b)
        # apply mask along time: h = h * x_mask
        h_masked = torch.empty_like(h)
        BLOCK_T = _pick_block_t(T)
        grid_mask = (B, half_channels, triton.cdiv(T, BLOCK_T))
        apply_mask_to_h[grid_mask](h, x_mask, h_masked, B=B, C1=half_channels, T=T, BLOCK_T=BLOCK_T)
        # update x1 = x1 + h (forward) or -h (reverse)
        # We need x1; split x = [x0, x1]
        x1 = x[:, half_channels:, :].contiguous()
        x1_out = torch.empty_like(x1)
        grid_add = (B, half_channels, triton.cdiv(T, BLOCK_T))
        add_h_to_x1[grid_add](x1, h_masked, x1_out, B=B, C1=half_channels, T=T, ADD=ADD, BLOCK_T=BLOCK_T)
        # concatenate [x0, x1_out]
        out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
        grid_first = (B, half_channels, triton.cdiv(T, BLOCK_T))
        concat_copy_first_half[grid_first](x0, out, B=B, C0=half_channels, C1=half_channels, T=T, BLOCK_T=BLOCK_T)
        grid_second = (B, half_channels, triton.cdiv(T, BLOCK_T))
        concat_copy_second_half[grid_second](x1_out, out, B=B, C0=half_channels, C1=half_channels, T=T, BLOCK_T=BLOCK_T)
        # apply x_mask across channels: out = out * x_mask (broadcast along channel)
        out_masked = torch.empty_like(out)
        grid_mask_out = (B, C, triton.cdiv(T, BLOCK_T))
        apply_mask_to_out_triton[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=BLOCK_T)
        return out_masked

    # Collect transforms (already on CUDA per get_inputs)
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

    # Ensure x is contiguous and float32
    x = x.contiguous().to(torch.float32)
    x_mask = x_mask.contiguous().to(torch.float32)

    if not reverse:
        # Forward: apply transforms sequentially
        x_out = x
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            x = apply_one_transform(x[:, :half_channels, :], conv0_w, conv0_b,
                                    conv1_w, conv1_b, conv2_w, conv2_b, ADD=True)
    else:
        # Reverse: apply in reverse order, subtract h
        x_out = x
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x = apply_one_transform(x[:, :half_channels, :], conv0_w, conv0_b,
                                    conv1_w, conv1_b, conv2_w, conv2_b, ADD=False)

    return x_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: x, x_mask, reverse, ... weights
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False
        return run(x, x_mask, reverse, *args[3:])


def run(*args):
    return ModelNew()(*args)
