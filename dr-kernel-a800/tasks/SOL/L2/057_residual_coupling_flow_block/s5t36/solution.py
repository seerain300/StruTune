import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _pick_block_t(T):
    # choose a reasonable block size for time dimension
    if T >= 2048:
        return 128
    elif T >= 512:
        return 128
    else:
        return 64


def _pick_num_warps(block_t):
    if block_t >= 128:
        return 4
    else:
        return 2


@triton.jit
def conv1d_stride1_bias_relu(
    x_ptr,         # *f32, [B, Cin, T]
    w_ptr,         # *f32, [Cout, Cin*K]
    b_ptr,         # *f32, [Cout]
    out_ptr,       # *f32, [B, Cout, T]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,      # kernel size (e.g., 5)
    PAD: tl.constexpr,    # padding = (K-1)//2
):
    # Each program handles one output (b, co), and we iterate over time t with BLOCK_T tiling
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # Accumulator for this (b, co, t_offsets)
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    # K is constexpr (e.g., 5), so Triton can unroll
    for c in tl.static_range(0, Cin):
        for k in tl.static_range(0, K):
            t_in = t_offsets + PAD - k  # vector of t_in
            in_range = (t_in >= 0) & (t_in < T)
            # pointer to x[b, c, t_in]
            x_index = (pid_b * (Cin * T)) + (c * T) + t_in
            x_val = tl.load(x_ptr + x_index, mask=mask_t & in_range, other=0.0)
            # pointer to w[co, c*K + k]
            w_index = pid_co * (Cin * K) + (c * K) + k
            w_val = tl.load(w_ptr + w_index)
            acc += x_val * w_val  # vector add

    # Add bias
    b_val = tl.load(b_ptr + pid_co)  # scalar
    acc = acc + b_val
    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to out[b, co, t_offsets]
    out_index = (pid_b * (Cout * T)) + (pid_co * T) + t_offsets
    tl.store(out_ptr + out_index, acc, mask=mask_t)


@triton.jit
def apply_mask_to_out_triton(
    out_ptr,       # *f32, [B, C, T]
    mask_ptr,      # *f32, [B, 1, T]
    out_ptr_out,   # *f32, [B, C, T]
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

    # mask is [B, 1, T]; index over t
    mask_index = (pid_b * T) + t_offsets
    mask_val = tl.load(mask_ptr + mask_index, mask=mask_t, other=1.0)

    out_val = out_val * mask_val
    tl.store(out_ptr_out + out_index, out_val, mask=mask_t)


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,        # *f32, [B, C1, T]
    h_ptr,         # *f32, [B, C1, T]
    out_ptr,       # *f32, [B, C1, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ADD: tl.constexpr,  # bool: True for add, False for subtract
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    h_index = x1_index
    out_index = x1_index

    x1_val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    h_val = tl.load(h_ptr + h_index, mask=mask_t, other=0.0)

    if ADD:
        res = x1_val + h_val
    else:
        res = x1_val - h_val

    tl.store(out_ptr + out_index, res, mask=mask_t)


@triton.jit
def concat_copy_first_half(
    x0_ptr,        # *f32, [B, C0, T]
    out_ptr,       # *f32, [B, C, T]
    B: tl.constexpr,
    C0: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # out[:, :C0, :] = x0[:, :, :]
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # pid_c in [0, C0)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x0_index = (((pid_b * C0) + pid_c) * T) + t_offsets
    out_index = (((pid_b * C) + pid_c) * T) + t_offsets

    val = tl.load(x0_ptr + x0_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


@triton.jit
def concat_copy_second_half(
    x1_ptr,        # *f32, [B, C1, T]
    out_ptr,       # *f32, [B, C, T]
    B: tl.constexpr,
    C1: tl.constexpr,
    C0: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # out[:, C0:C0+C1, :] = x1[:, :, :]
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)  # pid_c in [0, C1)
    pid_t_block = tl.program_id(2)
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    x1_index = (((pid_b * C1) + pid_c) * T) + t_offsets
    out_index = (((pid_b * C) + (C0 + pid_c)) * T) + t_offsets

    val = tl.load(x1_ptr + x1_index, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_index, val, mask=mask_t)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_mask, reverse, transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias, transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Triton-only forward that mirrors the original apply_transform behavior:
        - Forward: x1 = x1 + transform(x0) for each layer
        - Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
        """
        assert x.is_cuda and x_mask.is_cuda, "Tensors must be on CUDA for Triton"
        # Ensure dtype float32
        x = x.contiguous().to(torch.float32)
        x_mask = x_mask.contiguous().to(torch.float32)

        # Split input into two halves along channel dimension
        B, C, T = x.shape
        half_channels = C // 2
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

        # Define transforms
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

        # Forward or reverse pass
        if not reverse:
            # Forward: apply transforms sequentially
            # We will run for 4 transforms: each time h will be masked and added to x1
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # conv0: hidden_channels out, half_channels in
                y = torch.empty((B, conv0_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv0_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    x0, conv0_w, conv0_b, y,
                    B=B, Cin=half_channels, Cout=conv0_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                y_masked = torch.empty_like(y)
                grid_mask = (B, conv0_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](y, x_mask, y_masked, B=B, C=conv0_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # conv1: hidden_channels out, hidden_channels in
                h = torch.empty((B, conv1_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv1_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    y_masked, conv1_w, conv1_b, h,
                    B=B, Cin=conv0_w.shape[0], Cout=conv1_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                h_masked = torch.empty_like(h)
                grid_mask = (B, conv1_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](h, x_mask, h_masked, B=B, C=conv1_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # conv2: half_channels out, hidden_channels in
                h_out = torch.empty((B, conv2_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv2_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    h_masked, conv2_w, conv2_b, h_out,
                    B=B, Cin=conv1_w.shape[0], Cout=conv2_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                h_out_masked = torch.empty_like(h_out)
                grid_mask = (B, conv2_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](h_out, x_mask, h_out_masked, B=B, C=conv2_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # update x1: x1 = x1 + h_out (forward)
                x1_upd = torch.empty_like(x1)
                grid_update = (B, h_out_masked.shape[1], _ceil_div(T, _pick_block_t(T)))
                add_h_to_x1_triton[grid_update](
                    x1, h_out_masked, x1_upd,
                    B=B, C1=h_out_masked.shape[1], T=T, BLOCK_T=_pick_block_t(T), ADD=True,
                    num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # concatenate [x0, x1_upd] into out
                out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
                grid_first = (B, half_channels, _ceil_div(T, _pick_block_t(T)))
                concat_copy_first_half[grid_first](
                    x0, out, B=B, C0=half_channels, C=C, T=T, BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )
                grid_second = (B, h_out_masked.shape[0], _ceil_div(T, _pick_block_t(T)))
                concat_copy_second_half[grid_second](
                    x1_upd, out, B=B, C1=h_out_masked.shape[0], C0=half_channels, T=T, BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply final x_mask across channels
                out_masked = torch.empty_like(out)
                grid_mask_out = (B, C, T)
                apply_mask_to_out_triton[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=_pick_block_t(T))

                # set x for next iteration: x = [x0, x1_upd]
                x = out_masked

        else:
            # Reverse: apply in reverse order, subtract h
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # conv2: half_channels out, hidden_channels in
                y = torch.empty((B, conv2_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv2_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    x1, conv2_w, conv2_b, y,
                    B=B, Cin=conv1_w.shape[0], Cout=conv2_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                y_masked = torch.empty_like(y)
                grid_mask = (B, conv2_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](y, x_mask, y_masked, B=B, C=conv2_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # conv1: hidden_channels out, hidden_channels in
                h = torch.empty((B, conv1_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv1_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    y_masked, conv1_w, conv1_b, h,
                    B=B, Cin=conv0_w.shape[0], Cout=conv1_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                h_masked = torch.empty_like(h)
                grid_mask = (B, conv1_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](h, x_mask, h_masked, B=B, C=conv1_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # conv0: hidden_channels out, half_channels in
                h_out = torch.empty((B, conv0_w.shape[0], T), dtype=torch.float32, device=x.device)
                grid = (B, conv0_w.shape[0], _ceil_div(T, _pick_block_t(T)))
                conv1d_stride1_bias_relu[grid](
                    h_masked, conv0_w, conv0_b, h_out,
                    B=B, Cin=half_channels, Cout=conv0_w.shape[0], T=T, K=5, PAD=2,
                    BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply mask over channels
                h_out_masked = torch.empty_like(h_out)
                grid_mask = (B, conv0_w.shape[0], T)
                apply_mask_to_out_triton[grid_mask](h_out, x_mask, h_out_masked, B=B, C=conv0_w.shape[0], T=T, BLOCK_T=_pick_block_t(T))

                # update x1: x1 = x1 - h_out (reverse)
                x1_upd = torch.empty_like(x1)
                grid_update = (B, h_out_masked.shape[1], _ceil_div(T, _pick_block_t(T)))
                add_h_to_x1_triton[grid_update](
                    x1, h_out_masked, x1_upd,
                    B=B, C1=h_out_masked.shape[1], T=T, BLOCK_T=_pick_block_t(T), ADD=False,
                    num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # concatenate [x0, x1_upd] into out
                out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
                grid_first = (B, half_channels, _ceil_div(T, _pick_block_t(T)))
                concat_copy_first_half[grid_first](
                    x0, out, B=B, C0=half_channels, C=C, T=T, BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )
                grid_second = (B, h_out_masked.shape[0], _ceil_div(T, _pick_block_t(T)))
                concat_copy_second_half[grid_second](
                    x1_upd, out, B=B, C1=h_out_masked.shape[0], C0=half_channels, T=T, BLOCK_T=_pick_block_t(T), num_warps=_pick_num_warps(_pick_block_t(T)), num_stages=2
                )

                # apply final x_mask across channels
                out_masked = torch.empty_like(out)
                grid_mask_out = (B, C, T)
                apply_mask_to_out_triton[grid_mask_out](out, x_mask, out_masked, B=B, C=C, T=T, BLOCK_T=_pick_block_t(T))

                # set x for previous iteration: x = [x0, x1_upd]
                x = out_masked

        return x


# Helper functions for Triton kernel grids
def _ceil_div(a, b):
    return (a + b - 1) // b


def run(*args):
    return ModelNew()(*args)
