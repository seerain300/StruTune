import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel(
    x_ptr,                 # *float32, input x (B, L, H) contiguous
    w_ptr,                 # *float32, in_proj_weight (3H, H, L) contiguous
    b_ptr,                 # *float32, in_proj_bias (3H)
    out_ptr,               # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,               # sizes
    stride_x_b, stride_x_l, stride_x_h,     # strides for x (B, L, H)
    stride_w_j, stride_w_i, stride_w_l,     # strides for w (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l   # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    J = 3 * H
    j_offsets = j_block * 64 + tl.arange(0, 64)     # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)   # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulator for BCx[b, j, l]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel j, sum over input channels i and positions l
    for j in range(0, J):
        # Load bias for this j
        bias_val = tl.load(b_ptr + j)
        # Compute dot product over input channels i in [0..H-1]
        for i in range(0, H):
            # Load x[b, l, i]
            x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + i * stride_x_h
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (64,128)
            # Load weight[j, i, l]
            w_ptrs = w_ptr + j * stride_w_j + i * stride_w_i + l_offsets[None, :] * stride_w_l
            w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)
            # Accumulate
            acc += x_vals * w_vals
        # Add bias
        acc += bias_val

    # Store to out[b, j, l]
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Element-wise gating: out = a * b, a: (B, L, H), b: (B, L, H)
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


# 3) Grouped causal 1D convolution with kernel_size=4, stride=1, groups=H
# Input Bx: (B, H, L) contiguous (we will pass Bx_padded here as (B, H, L_padded))
# conv_weight: (H, H, 4) contiguous
# conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L_padded) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H, L_padded,     # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L_padded)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l # strides for out (B, H, L)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)     # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)   # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel ho (same as input channel hi due to groups=H), sum over its own input channel hi and kernel positions k
    for ho in range(0, H):
        # Initialize accumulator for this ho
        acc = tl.zeros((64, 128), dtype=tl.float32)
        for hi in range(0, H):
            # Sum over kernel positions k=0..3
            for k in range(0, 4):
                # Read Bx[b, hi, l + k] with causal indexing; out-of-range l+k due to padding are fine since we padded
                bx_ptrs = Bx_ptr + b * stride_Bx_b + hi * stride_Bx_h + (l_offsets[None, :] + k) * stride_Bx_l
                bx_vals = tl.load(bx_ptrs, mask=mask, other=0.0)  # (64,128)
                # Read conv_weight[ho, hi, k]
                w_ptrs = w_ptr + ho * stride_w_o + hi * stride_w_i + k * stride_w_k
                w_val = tl.load(w_ptrs)
                acc += bx_vals * w_val
        # Add conv bias
        bias_val = tl.load(bias_ptr + ho)
        acc += bias_val

        # Store conv_out[b, ho, l]
        out_ptrs = out_ptr + b * stride_out_b + ho * stride_out_h + l_offsets[None, :] * stride_out_l
        tl.store(out_ptrs, acc, mask=mask)


# 4) Final out-projection: output = F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, L, H), out_proj_weight: (H, L, H), out_proj_bias: (H), output: (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,                 # *float32, input y (B, L, H) contiguous
    w_ptr,                 # *float32, weight (H, L, H) contiguous
    b_ptr,                 # *float32, bias (H)
    out_ptr,               # *float32, output (B, L, H) contiguous
    B, L, H,               # sizes
    stride_y_b, stride_y_l, stride_y_h,     # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,     # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h   # strides for out (B, L, H)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)     # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)   # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h, accumulate over input channels hi and positions l
    for h in range(0, H):
        for hi in range(0, H):
            # y[b, l, hi]
            y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hi * stride_y_h
            y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)
            # weight[h, l, hi] -> w_ptr + h * stride_w_o + l_offsets * stride_w_l + hi * stride_w_i
            w_ptrs = w_ptr + h * stride_w_o + l_offsets[None, :] * stride_w_l + hi * stride_w_i
            w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)
            acc += y_vals * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store output[b, l, h]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure inputs are float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, L, H = x.shape
        J = 3 * H

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, J, L), dtype=torch.float32, device=x.device)
        in_proj_kernel[(triton.cdiv(J, 64), triton.cdiv(L, 128), B)](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2)
        )

        # 2) Transpose for chunking (PyTorch view is fine here; we do not use torch.chunk)
        # BCx has shape (B, 3H, L). We need (B, L, 3H) and then split into (B, L, H) parts:
        BCx_T = BCx.transpose(-1, -2).contiguous()  # shape (B, L, 3H)
        B_tensor = BCx_T[:, :, 0:H].contiguous()   # (B, L, H)
        C_tensor = BCx_T[:, :, H:2*H].contiguous() # (B, L, H)
        x_proj = BCx_T[:, :, 2*H:3*H].contiguous() # (B, L, H)

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        gate_mul_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2)
        )

        # 4) Causal padding on sequence dimension by K-1 = 3
        # Since conv kernel_size=4, stride=1, causal => pad left by 3
        L_padded = L + 3
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad on last dim: left=3, right=0 -> (B, H, L+3)
        Bx_padded = Bx_padded.contiguous()

        # 5) Grouped causal conv: conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        conv1d_grouped_causal_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, L, H, L_padded,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)
        )

        # 6) Output gating: y = C * conv_out
        y = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        gate_mul_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            C_tensor, conv_out.transpose(-1, -2).contiguous(), y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.transpose(-1, -2).contiguous().stride(0), conv_out.transpose(-1, -2).contiguous().stride(1), conv_out.transpose(-1, -2).contiguous().stride(2),
            y.stride(0), y.stride(1), y.stride(2)
        )

        # 7) Final out-projection: output (B, L, H)
        output = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        out_proj_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            y, out_proj_weight, out_proj_bias, output,
            B, L, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2)
        )

        return output


def run(*args):
    return ModelNew()(*args)
