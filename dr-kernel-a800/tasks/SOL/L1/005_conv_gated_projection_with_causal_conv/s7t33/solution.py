import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: x -> BCx of shape (B, 3H, L)
# x: (B, L, H), in_proj_weight: (3H, H, L), bias: (3H,)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                   # *float32, input x (B, L, H)
    weight_ptr,              # *float32, in_proj_weight (3H, H, L)
    bias_ptr,                # *float32, in_proj_bias (3H,)
    out_ptr,                 # *float32, output BCx (B, 3H, L)
    B, L, H,                 # sizes
    stride_x_b, stride_x_l, stride_x_h,      # strides for x (B, L, H)
    stride_w_o, stride_w_i, stride_w_l,      # strides for weight (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, B, L_tiles)
    j_block = tl.program_id(0)  # over 3H channels
    b = tl.program_id(1)
    l_block = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64], corresponds to 3H
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128], along sequence length

    mask_j = j_offsets < (3 * H)
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Compute output pointers
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l

    # Load corresponding weight row j and bias
    w_j_ptr = weight_ptr + j_offsets * stride_w_o
    bias_j = tl.load(bias_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over input channels H and sequence L to accumulate F.linear
    # x[b, l, h], weight[j, h, l] -> out[b, j, l]
    for ho in range(0, H):
        # Load x[b, l, ho]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load weight[j, ho, l]
        w_ptrs = w_j_ptr + ho * stride_w_i + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += x_vals * w_vals

    # Add bias
    acc += bias_j[:, None]

    # Store
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk BCx (B, 3H, L) into B, C, X_proj (B, L, H) via Triton
# We pass pointers and strides; kernel copies slices along last dim of size H.
@triton.jit
def chunk3_kernel(
    inp_ptr,                 # *float32, input BCx (B, 3H, L) contiguous
    outB_ptr, outC_ptr, outX_ptr,  # *float32, outputs (B, L, H) contiguous
    B, L, H,                 # sizes
    stride_inp_b, stride_inp_j, stride_inp_l,    # strides for inp (B, 3H, L)
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
        in_ptrs_x = inp_ptr + b * stride_inp_b + ho * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_x = tl.load(in_ptrs_x, mask=mask, other=0.0)  # (64,128)
        out_ptrs_x = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_x, vals_x, mask=mask)

    # Copy B from inp[b, l, H..2H-1]
    for ho in range(0, H):
        in_ptrs_B = inp_ptr + b * stride_inp_b + (H + ho) * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_B = tl.load(in_ptrs_B, mask=mask, other=0.0)  # (64,128)
        out_ptrs_B = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, vals_B, mask=mask)

    # Copy C from inp[b, l, 2H..3H-1]
    for ho in range(0, H):
        in_ptrs_C = inp_ptr + b * stride_inp_b + (2 * H + ho) * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_C = tl.load(in_ptrs_C, mask=mask, other=0.0)  # (64,128)
        out_ptrs_C = outC_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_C, vals_C, mask=mask)


# 3) Element-wise gating: out = a * b (vectorized)
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
# Output conv_out: (B, H, L), stride=1, padding=K-1=3, groups=H (depthwise)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l  # strides for out (B, H, L)
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

    # For grouped conv, each output channel h accumulates over its own input channel h and kernel positions
    for ho in range(0, H):
        # weight w[ho, ho, k] scalar (since groups=H)
        # Accumulate over k in [0,4)
        for k in range(0, 4):
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            # input Bx[b, ho, l]
            in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_offsets[None, :] * stride_Bx_l
            in_vals = tl.load(in_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along ho
            acc += w_val * in_vals

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Final out-projection: Y (B, L, H) -> out (B, L, H) with weight (H, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,     # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_h,     # strides for w (H, L, H)
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

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over weight channels o in [0, H)
    for ho in range(0, H):
        # weight[ho, l, h]
        w_ptrs = w_ptr + ho * stride_w_o + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        # y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along ho

        acc += w_vals * y_vals

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Shapes: x (B, L, H), in_proj_weight (3H, H, L), conv_weight (H, H, 4), out_proj_weight (H, L, H)
        B, L, H = x.shape

        # 1) In-projection: compute BCx (B, 3H, L) via Triton
        BCx = torch.empty((B, 3 * H, L), dtype=torch.float32, device=x.device)
        x_c = x.contiguous().to(torch.float32)
        w_c = in_proj_weight.contiguous().to(torch.float32)
        b_c = in_proj_bias.contiguous().to(torch.float32)

        grid_in = (triton.cdiv(3 * H, 64), B, triton.cdiv(L, 128))
        in_proj_kernel_B[grid_in](
            x_c, w_c, b_c, BCx,
            B, L, H,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Chunk BCx into B, C, X_proj (each B, L, H) via Triton
        B_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        C_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        X_proj = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        BCx_c = BCx.contiguous()
        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_c, B_tensor, C_tensor, X_proj,
            B, L, H,
            BCx_c.stride(0), BCx_c.stride(1), BCx_c.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B * X_proj
        Bx = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        grid_mul = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_mul](
            B_tensor, X_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            X_proj.stride(0), X_proj.stride(1), X_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D conv: Bx (B, H, L) -> conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        Bx_c = Bx.contiguous().transpose(1, 2).contiguous()  # (B, L, H) -> (B, H, L) logically, but we use (B, L, H)
        # Note: In the Triton kernel, we treat Bx as (B, H, L) by indexing appropriately. We pass Bx_c as (B, L, H) and compute (B, H, L) out.
        w_c = conv_weight.contiguous().to(torch.float32)  # (H, H, 4)
        b_c = conv_bias.contiguous().to(torch.float32)    # (H,)

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_c, w_c, b_c, conv_out,
            B, L, H,
            Bx_c.stride(0), Bx_c.stride(1), Bx_c.stride(2),   # for input (B, L, H), but we index as (B, H, L) in kernel
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Output gating: Y = C * conv_out
        Y = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        grid_mul2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_mul2](
            C_tensor, conv_out, Y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2),
            num_warps=4, num_stages=2
        )

        # 6) Final out-projection: Y (B, L, H) via weight (H, L, H)
        # Note: out_proj_weight is provided as (H, L, H). We need to compute F.linear(Y, out_proj_weight, out_proj_bias).
        Y_T = Y.transpose(1, 2).contiguous()  # (B, L, H)
        output = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        w_c_out = out_proj_weight.contiguous().to(torch.float32)  # (H, L, H)
        b_out = out_proj_bias.contiguous().to(torch.float32)      # (H,)

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            Y_T, w_c_out, b_out, output,
            B, L, H,
            Y_T.stride(0), Y_T.stride(1), Y_T.stride(2),
            w_c_out.stride(0), w_c_out.stride(1), w_c_out.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
