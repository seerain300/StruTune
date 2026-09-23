import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_stride1_bias_relu(
    x_ptr,          # *f32, input [B, Cin, T]
    w_ptr,          # *f32, weights [Cout, Cin*K]
    b_ptr,          # *f32, bias [Cout]
    out_ptr,        # *f32, output [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    PAD: tl.constexpr,   # for K=5, PAD=2
    K: tl.constexpr,     # kernel size, e.g., 5
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)   # output channel index
    pid_t_block = tl.program_id(2)  # tile over time

    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # accumulate over input channels and kernel taps
    for ci in range(Cin):
        for k in tl.static_range(K):
            t_in = t_offsets - PAD + k
            in_range = (t_in >= 0) & (t_in < T) & mask_t
            x_index = pid_b * (Cin * T) + ci * T + t_in
            x_val = tl.load(x_ptr + x_index, mask=in_range, other=0.0)
            w_index = pid_co * (Cin * K) + ci * K + k
            w_val = tl.load(w_ptr + w_index)
            acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + pid_co)
    acc = acc + b_val

    # apply ReLU
    acc = tl.maximum(acc, 0.0)

    # store output
    out_index = pid_b * (Cout * T) + pid_co * T + t_offsets
    tl.store(out_ptr + out_index, acc, mask=mask_t)


@triton.jit
def ones_mask_triton(
    mask_ptr,   # *f32, [B, 1, T]
    B: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t_block = tl.program_id(1)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T
    # mask_ptr indexing: [B, 1, T] -> index = b*T + t
    idx = pid_b * T + t_offsets
    ones = tl.full([BLOCK_T], 1.0, dtype=tl.float32)
    tl.store(mask_ptr + idx, ones, mask=mask_t)


@triton.jit
def apply_mask_to_h_triton(
    h_ptr,      # *f32, [B, Cout, T]
    mask_ptr,   # *f32, [B, 1, T]
    h_out_ptr,  # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    h_index = pid_b * (Cout * T) + pid_co * T + t_offsets
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    # mask is [B, 1, T]
    mask_index = pid_b * T + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    h_val = h_val * mask_val
    tl.store(h_out_ptr + h_index, h_val, mask=mask_t)


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,     # *f32, [B, C1, T]
    h_ptr,      # *f32, [B, C1, T]
    out_ptr,    # *f32, [B, C1, T]
    ADD: tl.constexpr,     # True: add, False: subtract
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

    x1_index = (pid_b * C1 + pid_c) * T + t_offsets
    h_index = x1_index
    val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)
    if ADD:
        val = val + h_val
    else:
        val = val - h_val
    tl.store(out_ptr + x1_index, val, mask=mask_t)


@triton.jit
def concat_copy_first_half(
    x0_ptr,      # *f32, [B, C0, T]
    out_ptr,     # *f32, [B, C, T], C >= C0
    B: tl.constexpr,
    C0: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [0, C0)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    src_index = pid_b * (C0 * T) + pid_c * T + t_offsets
    dst_index = pid_b * (C0 * T) + pid_c * T + t_offsets  # out[:, :C0, :]
    val = tl.load(x0_ptr + src_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + dst_index, val, mask=mask_t)


@triton.jit
def concat_copy_second_half(
    x1_ptr,      # *f32, [B, C1, T]
    out_ptr,     # *f32, [B, C, T], C >= C0+C1
    C0: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # c in [0, C1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    src_index = pid_b * (C1 * T) + pid_c * T + t_offsets
    dst_index = pid_b * ((C0 + C1) * T) + (pid_c + C0) * T + t_offsets
    val = tl.load(x1_ptr + src_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + dst_index, val, mask=mask_t)


def _pick_block_t(T):
    if T >= 4096:
        return 256
    elif T >= 2048:
        return 128
    elif T >= 1024:
        return 128
    else:
        return 64


class ModelNew(nn.Module):
    def forward(self, x, x_mask, reverse, transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias, transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        # Ensure CUDA and dtype float32
        x = x.contiguous().to(torch.float32)
        B, C, T = x.shape
        half_channels = C // 2

        # Generate x_mask via Triton (ones over [B, 1, T])
        x_mask = x_mask.contiguous()
        x_mask = x_mask.to(torch.float32)
        # If x_mask is not provided as [B,1,T], create with Triton
        if x_mask.shape != (B, 1, T):
            x_mask = torch.empty((B, 1, T), dtype=torch.float32, device=x.device)
            grid_mask = (B, _pick_block_t(T))
            ones_mask_triton[grid_mask](x_mask, B=B, T=T, BLOCK_T=128)

        # Prepare transforms: ensure on CUDA and float32
        # conv weights are [Cout, Cin, K]; we will pass as [Cout, Cin*K] and extract indices accordingly
        # Create transforms list of tuples (w0, b0, w1, b1, w2, b2)
        transforms = []
        # Transform 0
        w0 = transform_0_conv0_weight.contiguous().to(torch.float32)
        b0 = transform_0_conv0_bias.contiguous().to(torch.float32)
        w1 = transform_0_conv1_weight.contiguous().to(torch.float32)
        b1 = transform_0_conv1_bias.contiguous().to(torch.float32)
        w2 = transform_0_conv2_weight.contiguous().to(torch.float32)
        b2 = transform_0_conv2_bias.contiguous().to(torch.float32)
        transforms.append((w0, b0, w1, b1, w2, b2))

        # Transform 1
        w0 = transform_1_conv0_weight.contiguous().to(torch.float32)
        b0 = transform_1_conv0_bias.contiguous().to(torch.float32)
        w1 = transform_1_conv1_weight.contiguous().to(torch.float32)
        b1 = transform_1_conv1_bias.contiguous().to(torch.float32)
        w2 = transform_1_conv2_weight.contiguous().to(torch.float32)
        b2 = transform_1_conv2_bias.contiguous().to(torch.float32)
        transforms.append((w0, b0, w1, b1, w2, b2))

        # Transform 2
        w0 = transform_2_conv0_weight.contiguous().to(torch.float32)
        b0 = transform_2_conv0_bias.contiguous().to(torch.float32)
        w1 = transform_2_conv1_weight.contiguous().to(torch.float32)
        b1 = transform_2_conv1_bias.contiguous().to(torch.float32)
        w2 = transform_2_conv2_weight.contiguous().to(torch.float32)
        b2 = transform_2_conv2_bias.contiguous().to(torch.float32)
        transforms.append((w0, b0, w1, b1, w2, b2))

        # Transform 3
        w0 = transform_3_conv0_weight.contiguous().to(torch.float32)
        b0 = transform_3_conv0_bias.contiguous().to(torch.float32)
        w1 = transform_3_conv1_weight.contiguous().to(torch.float32)
        b1 = transform_3_conv1_bias.contiguous().to(torch.float32)
        w2 = transform_3_conv2_weight.contiguous().to(torch.float32)
        b2 = transform_3_conv2_bias.contiguous().to(torch.float32)
        transforms.append((w0, b0, w1, b1, w2, b2))

        # Initialize x_out
        x_out = x
        C0 = half_channels
        C1 = half_channels

        if not reverse:
            # Forward pass
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # x0: first half channels
                x0 = x_out[:, :C0, :].contiguous()
                # conv0
                Cout0 = conv0_w.shape[0]
                w0 = conv0_w.contiguous()
                b0 = conv0_b.contiguous()
                h0 = torch.empty((B, Cout0, T), dtype=torch.float32, device=x.device)
                grid0 = (B, Cout0, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid0](x0, w0, b0, h0, B=B, Cin=C0, Cout=Cout0, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)
                # ReLU already applied in the kernel

                # conv1
                Cout1 = conv1_w.shape[0]
                w1 = conv1_w.contiguous()
                b1 = conv1_b.contiguous()
                h1 = torch.empty((B, Cout1, T), dtype=torch.float32, device=x.device)
                grid1 = (B, Cout1, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid1](h0, w1, b1, h1, B=B, Cin=Cout0, Cout=Cout1, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)

                # conv2
                Cout2 = conv2_w.shape[0]
                w2 = conv2_w.contiguous()
                b2 = conv2_b.contiguous()
                h = torch.empty((B, Cout2, T), dtype=torch.float32, device=x.device)
                grid2 = (B, Cout2, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid2](h1, w2, b2, h, B=B, Cin=Cout1, Cout=Cout2, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)

                # apply mask
                h_masked = torch.empty_like(h)
                grid_mask_h = (B, Cout2, T)
                apply_mask_to_h_triton[grid_mask_h](h, x_mask, h_masked, B=B, Cout=Cout2, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # update x1
                x1 = x_out[:, C0:, :].contiguous()
                x1_out = torch.empty_like(x1)
                grid_add = (B, C1, _ceil_div(T, 128))
                add_h_to_x1_triton[grid_add](x1, h_masked, x1_out, ADD=True, B=B, C1=C1, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # concatenate
                out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
                grid_first = (B, C0, _ceil_div(T, 128))
                concat_copy_first_half[grid_first](x0, out, B=B, C0=C0, T=T, BLOCK_T=128, num_warps=4, num_stages=2)
                grid_second = (B, C1, _ceil_div(T, 128))
                concat_copy_second_half[grid_second](x1_out, out, C0=C0, C1=C1, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # apply final mask (broadcast along channels)
                out_masked = torch.empty_like(out)
                grid_mask_out = (B, C, T)
                apply_mask_to_out_triton[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                x_out = out_masked
        else:
            # Reverse pass: apply in reverse order, subtract
            # We reuse the same structure; only change ADD=False in add_h_to_x1_triton and iterate backwards
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # x0: first half channels
                x0 = x_out[:, :C0, :].contiguous()
                # conv0
                Cout0 = conv0_w.shape[0]
                w0 = conv0_w.contiguous()
                b0 = conv0_b.contiguous()
                h0 = torch.empty((B, Cout0, T), dtype=torch.float32, device=x.device)
                grid0 = (B, Cout0, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid0](x0, w0, b0, h0, B=B, Cin=C0, Cout=Cout0, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)

                # conv1
                Cout1 = conv1_w.shape[0]
                w1 = conv1_w.contiguous()
                b1 = conv1_b.contiguous()
                h1 = torch.empty((B, Cout1, T), dtype=torch.float32, device=x.device)
                grid1 = (B, Cout1, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid1](h0, w1, b1, h1, B=B, Cin=Cout0, Cout=Cout1, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)

                # conv2
                Cout2 = conv2_w.shape[0]
                w2 = conv2_w.contiguous()
                b2 = conv2_b.contiguous()
                h = torch.empty((B, Cout2, T), dtype=torch.float32, device=x.device)
                grid2 = (B, Cout2, _ceil_div(T, 128))
                conv1d_stride1_bias_relu[grid2](h1, w2, b2, h, B=B, Cin=Cout1, Cout=Cout2, T=T, PAD=2, K=5, BLOCK_T=128, num_warps=4, num_stages=2)

                # apply mask
                h_masked = torch.empty_like(h)
                grid_mask_h = (B, Cout2, T)
                apply_mask_to_h_triton[grid_mask_h](h, x_mask, h_masked, B=B, Cout=Cout2, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # update x1 (subtract)
                x1 = x_out[:, C0:, :].contiguous()
                x1_out = torch.empty_like(x1)
                grid_add = (B, C1, _ceil_div(T, 128))
                add_h_to_x1_triton[grid_add](x1, h_masked, x1_out, ADD=False, B=B, C1=C1, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # concatenate
                out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
                grid_first = (B, C0, _ceil_div(T, 128))
                concat_copy_first_half[grid_first](x0, out, B=B, C0=C0, T=T, BLOCK_T=128, num_warps=4, num_stages=2)
                grid_second = (B, C1, _ceil_div(T, 128))
                concat_copy_second_half[grid_second](x1_out, out, C0=C0, C1=C1, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                # apply final mask (broadcast along channels)
                out_masked = torch.empty_like(out)
                grid_mask_out = (B, C, T)
                apply_mask_to_out_triton[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=128, num_warps=4, num_stages=2)

                x_out = out_masked

        return x_out


# Utility for ceil-div (not in triton)
def _ceil_div(a, b):
    return (a + b - 1) // b


def run(*args):
    return ModelNew()(*args)
