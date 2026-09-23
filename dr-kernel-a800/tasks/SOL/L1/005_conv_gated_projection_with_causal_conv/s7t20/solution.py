import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx of shape (B, 3H, L)
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H,)
# Output BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                  # *float32, input x (B, L, H) contiguous
    w_ptr,                  # *float32, weight (3H, H, L) contiguous
    bias_ptr,               # *float32, bias (3H,)
    BCx_ptr,                # *float32, output (B, 3H, L) contiguous
    B, L, H,                # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_o, stride_w_i, stride_w_l,   # strides for w
    stride_BCx_b, stride_BCx_o, stride_BCx_l   # strides for BCx
):
    # Grid: (O_tiles, L_tiles, B), where O = 3H
    o_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    O = 3 * H

    o_offsets = o_block * 64 + tl.arange(0, 64)   # [64], index over 3H
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128], index over L

    mask_o = o_offsets < O
    mask_l = l_offsets < L
    mask = mask_o[:, None] & mask_l[None, :]

    # Accumulator for BCx[b, o, l]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Load corresponding input channel j = o % H and input sequence l
    # For each output o in [0, 3H), input feature index j = o % H
    # x[b, l, j] -> j = o % H
    for o in range(0, 64):  # vectorized over 64 outputs at a time
        j = o_offsets[o] % H
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + j * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast for o
        # Load weight w[o, j, l] for this tile
        w_ptrs = w_ptr + o * stride_w_o + j * stride_w_i + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)
        acc[o, :] = x_vals * w_vals

    # Add bias
    bias_o = o_offsets % (3 * H)  # o_offsets already in range, bias is (3H,)
    bias_vals = tl.load(bias_ptr + bias_o, mask=mask_o, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to BCx[b, o, l]
    BCx_ptrs = BCx_ptr + b * stride_BCx_b + o_offsets[:, None] * stride_BCx_o + l_offsets[None, :] * stride_BCx_l
    tl.store(BCx_ptrs, acc, mask=mask)


# 2) Chunking: split BCx (B, 3H, L) -> three (B, H, L): B_tensor, C_tensor, x_proj
@triton.jit
def chunk3_kernel(
    inp_ptr,                # *float32, input BCx (B, 3H, L) contiguous
    outB_ptr, outC_ptr, outX_ptr,  # outputs (B, L, H) contiguous
    B, L, H,                # sizes
    stride_inp_b, stride_inp_o, stride_inp_l,    # strides for inp (B, 3H, L)
    stride_out_b, stride_out_l, stride_out_h     # strides for outputs (B, L, H)
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

    # Copy x_proj from inp[b, l, 0..H-1]
    for ho in range(0, H):
        in_ptrs_x = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + ho * stride_inp_o
        vals_x = tl.load(in_ptrs_x, mask=mask, other=0.0)  # (64,128)
        out_ptrs_x = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_x, vals_x, mask=mask)

    # Copy B from inp[b, l, H..2H-1]
    for ho in range(0, H):
        in_ptrs_B = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + (H + ho) * stride_inp_o
        vals_B = tl.load(in_ptrs_B, mask=mask, other=0.0)  # (64,128)
        out_ptrs_B = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, vals_B, mask=mask)

    # Copy C from inp[b, l, 2H..3H-1]
    for ho in range(0, H):
        in_ptrs_C = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + (2 * H + ho) * stride_inp_o
        vals_C = tl.load(in_ptrs_C, mask=mask, other=0.0)  # (64,128)
        out_ptrs_C = outC_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_C, vals_C, mask=mask)


# 3) Element-wise gating: out = a * b (vectorized), a: (B, L, H), b: (B, L, H)
@triton.jit
def gate_mul_kernel(
    a_ptr, b_ptr, out_ptr,   # pointers
    B, L, H,                 # sizes
    stride_a_b, stride_a_l, stride_a_h,    # strides for a (B, L, H)
    stride_b_b, stride_b_l, stride_b_h,    # strides for b (B, L, H)
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

    a_ptrs = a_ptr + b * stride_a_b + l_offsets[None, :] * stride_a_l + h_offsets[:, None] * stride_a_h
    b_ptrs = b_ptr + b * stride_b_b + l_offsets[None, :] * stride_b_l + h_offsets[:, None] * stride_b_h
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h

    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    out_vals = a_vals * b_vals
    tl.store(out_ptrs, out_vals, mask=mask)


# 4) Grouped causal 1D convolution:
# Input Bx: (B, H, L), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_ho, stride_w_hi, stride_w_k,    # strides for weight (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l   # strides for out (B, H, L)
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

    # K = 4 for causal kernel
    K = 4
    for ho in range(0, H):
        # Load Bx[b, ho, l]
        Bx_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_offsets[None, :] * stride_Bx_l
        bx_vals = tl.load(Bx_ptrs, mask=mask, other=0.0)  # (64,128)

        # Accumulate over kernel positions
        for k in range(0, K):
            # weight w[ho, ho, k] scalar since groups=H
            w_ptrs = w_ptr + ho * stride_w_ho + ho * stride_w_hi + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            acc += bx_vals * w_val

    # Add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Final out-projection:
# y: (B, L, H), weight: (H, L, H), bias: (H), output: (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,       # strides for y (B, L, H)
    stride_w_h, stride_w_l, stride_w_ho,      # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h  # strides for out (B, L, H)
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

    # y[b, l, h] and w[h, l, ho] multiply
    acc = tl.zeros((64, 128), dtype=tl.float32)

    for ho in range(0, H):
        # y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)
        # w[ho, l, h] -> broadcast w over l, vector over ho
        w_ptrs = w_ptr + ho * stride_w_h + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_ho
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)
        acc += y_vals * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store output[b, l, h]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
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
        # Ensure contiguity and dtype
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, L, H = x.shape

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
        )

        # 2) Chunk BCx -> B_tensor, C_tensor, x_proj (B, L, H)
        B_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        C_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        x_proj = torch.empty((B, L, H), device=x.device, dtype=torch.float32)

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx, B_tensor, C_tensor, x_proj,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
        )

        # 4) Grouped causal 1D conv: conv_out (B, H, L)
        # Pad Bx on sequence dim by K-1 for causal
        pad = conv_weight.shape[2] - 1  # 4 - 1 = 3
        Bx_padded = nn.functional.pad(Bx.transpose(-1, -2), (pad, 0)).transpose(-1, -2).contiguous()  # shape (B, H, L + pad)
        conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        )

        # 5) Output gating: y = C_tensor * conv_out (shape (B, H, L))
        y = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        # For element-wise multiply, launch gate_mul_kernel again
        gate_mul_kernel[grid_out](
            C_tensor, conv_out, y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
        )

        # 6) Final out-projection: (B, L, H)
        output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_last = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_last](
            y.transpose(-1, -2).contiguous(), out_proj_weight, out_proj_bias, output,
            B, L, H,
            y.transpose(-1, -2).contiguous().stride(0), y.transpose(-1, -2).contiguous().stride(1), y.transpose(-1, -2).contiguous().stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )

        return output


def run(*args):
    return ModelNew()(*args)
