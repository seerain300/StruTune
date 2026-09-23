import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# 1) In-projection: compute BCx of shape (B, 3H, L) via Triton
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                  # *float32, x (B, L, H) contiguous
    w_ptr,                  # *float32, weight (3H, H, L) contiguous
    b_ptr,                  # *float32, bias (3H)
    out_ptr,                # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,                # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x (B, L, H)
    stride_w_j, stride_w_h, stride_w_l,   # strides for w (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l   # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, L_tiles, B), where J = 3H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)     # [64] over 3H
    l_offsets = l_block * 128 + tl.arange(0, 128)   # [128] over L

    mask_j = j_offsets < (3 * H)
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # For each j in [0, 3H), compute out[b, j, l] = sum_h w[j, h, l] * x[b, l, h] + b[j]
    for ho in range(0, H):  # loop over h contributes to each j
        # w[j, ho, l_offsets]
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + ho * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        # x[b, l_offsets, ho]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along j

        # pointer for out[b, j, l]
        out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
        # accumulate: out[b, j, l] = sum_l w_vals * x_vals
        # initialize output tile with zero
        tl.store(out_ptrs, tl.zeros((64, 128), dtype=tl.float32), mask=mask)

        # add contribution: out[b, j, l] += w_vals * x_vals
        # Note: since w_vals and x_vals broadcast along j and l, we need to sum over h explicitly.
        # To do that, we loop over h here:
        # out[b, j, l] += w[j, h, l] * x[b, l, h]
        # We'll replace the above zero-store with an accumulate loop to avoid overwriting.
        # Implement proper accumulation:
        acc = tl.zeros((64, 128), dtype=tl.float32)
        for ho2 in range(0, H):
            w_ptrs2 = w_ptr + j_offsets[:, None] * stride_w_j + ho2 * stride_w_h + l_offsets[None, :] * stride_w_l
            w_vals2 = tl.load(w_ptrs2, mask=mask, other=0.0)
            x_ptrs2 = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho2 * stride_x_h
            x_vals2 = tl.load(x_ptrs2, mask=mask_l[None, :], other=0.0)
            acc += w_vals2 * x_vals2

        # add bias for j
        bias_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
        bias_broadcast = bias_vals[:, None]
        out_ptrs_bias = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
        out_vals = tl.load(out_ptrs_bias, mask=mask, other=0.0) + bias_broadcast
        tl.store(out_ptrs_bias, out_vals, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H) via Triton
# F.linear(y, out_proj_weight, out_proj_bias) where out_proj_weight is (H, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H) contiguous
    w_ptr,               # *float32, weight (H, L, H) contiguous
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H) contiguous
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,    # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h   # strides for out (B, L, H)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h, accumulate: out[b, l, h] = sum_i w[h, l, i] * y[b, l, i] + b[h]
    for hi in range(0, H):
        # y[b, l_offsets, hi]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hi * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)

        # w[hi, l_offsets, hi] vector over l
        w_ptrs = w_ptr + hi * stride_w_o + l_offsets[None, :] * stride_w_l + hi * stride_w_i
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += w_vals * y_vals

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # Shapes: x (B, L, H), in_proj_weight (3H, H, L), in_proj_bias (3H),
        # conv_weight (H, H, 4), conv_bias (H), out_proj_weight (H, L, H), out_proj_bias (H)
        assert x.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda
        B, L, H = x.shape
        device = x.device
        dtype = torch.float32

        # Make inputs contiguous


def run(*args):
    return ModelNew()(*args)
