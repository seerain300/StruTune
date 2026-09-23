import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx = F.linear(x_transposed, in_proj_weight, in_proj_bias)
# x_transposed: (B, H, L); in_proj_weight: (3H, H, L); BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                # *float32, input x (B, H, L) contiguous
    w_ptr,                # *float32, weight (3H, H, L) contiguous
    bias_ptr,             # *float32, bias (3H)
    BCx_ptr,              # *float32, output BCx (B, 3H, L) contiguous
    B, H, L,              # sizes
    stride_x_b, stride_x_h, stride_x_l,   # strides for x (B, H, L)
    stride_w_o, stride_w_i, stride_w_l,   # strides for weight (3H, H, L)
    stride_BC_b, stride_BC_o, stride_BC_l  # strides for BCx (B, 3H, L)
):
    # Grid: (O_tiles, L_tiles, B) where O=3H
    O = 3 * H  # out_features
    o_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    o_offsets = o_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_o = o_offsets < O
    mask_l = l_offsets < L
    mask = mask_o[:, None] & mask_l[None, :]

    # Compute in_features-wise reduction: for each i in [0..H-1] and l, accumulate over O
    # out[b, o, l] = sum_i x[b, i, l] * w[o, i, l] + bias[o]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Reduction over i (in_features = H)
    for i in range(0, H):
        # x[b, i, l]
        x_ptrs = x_ptr + b * stride_x_b + i * stride_x_h + l_offsets[None, :] * stride_x_l  # (1,128)
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)

        # w[o, i, l] -> vector over o
        w_ptrs = w_ptr + o_offsets[:, None] * stride_w_o + i * stride_w_i + l_offsets[None, :] * stride_w_l  # (64,128)
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)

        acc += x_vals * w_vals  # broadcast x_vals along o

    # add bias
    bias_vals = tl.load(bias_ptr + o_offsets, mask=mask_o, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # store to BCx[b, o, l]
    out_ptrs = BCx_ptr + b * stride_BC_b + o_offsets[:, None] * stride_BC_o + l_offsets[None, :] * stride_BC_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk: split BCx (B, 3H, L) -> B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
# We transpose to (B, L, 3H) and copy slices to outputs using Triton.
@triton.jit
def chunk3_kernel(
    BCx_ptr,        # *float32, input BCx (B, L, 3H) contiguous
    outB_ptr,       # *float32, output B (B, L, H) contiguous
    outC_ptr,       # *float32, output C (B, L, H) contiguous
    outX_ptr,       # *float32, output x_proj (B, L, H) contiguous
    B, L, H,        # sizes
    stride_BC_b, stride_BC_l, stride_BC_o,   # strides for BCx (B, L, 3H)
    stride_out_b, stride_out_l, stride_out_h  # strides for outputs (B, L, H)
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

    # B: slice o = 0..H-1
    inB_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + h_offsets[:, None] * stride_BC_o
    outB_ptrs = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outB_ptrs, tl.load(inB_ptrs, mask=mask, other=0.0), mask=mask)

    # C: slice o = H..2H-1
    inC_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + (H + h_offsets[:, None]) * stride_BC_o
    outC_ptrs = outC_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outC_ptrs, tl.load(inC_ptrs, mask=mask, other=0.0), mask=mask)

    # x_proj: slice o = 2H..3H-1
    inX_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + (2 * H + h_offsets[:, None]) * stride_BC_o
    outX_ptrs = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outX_ptrs, tl.load(inX_ptrs, mask=mask, other=0.0), mask=mask)


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
# Output conv_out: (B, H, L)
# Implement padding in-kernel for causal: pad left by 3 -> l_in = l + 3
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l # strides for out (B, H, L)
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

    # Sum over K=4 positions with causal padding
    for k in range(0, 4):
        l_in = l_offsets + (3 - k)  # left causal pad
        # mask for valid input positions
        valid = l_in >= 0
        l_in = tl.where(valid, l_in, 0)  # avoid negative indexing in load

        # Read Bx[b, h, l_in]
        bx_ptrs = Bx_ptr + b * stride_Bx_b + h_offsets[:, None] * stride_Bx_h + l_in * stride_Bx_l
        bx_vals = tl.load(bx_ptrs, mask=(mask & valid), other=0.0)  # (64,128)

        # weight w[h, h, k] scalar (groups=H => output channel equals input channel)
        w_vals = tl.load(w_ptr + h_offsets * stride_w_o + h_offsets * stride_w_i + k * stride_w_k)  # (64,)

        # Accumulate
        acc += w_vals[:, None] * bx_vals

    # Add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Final out-projection: y (B, L, H) and weight (H, L, H) -> output (B, L, H)
# F.linear(y, out_proj_weight, out_proj_bias): out[l, h] = sum_i y[l, i] * w[i, l, h] + bias[h]
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H) contiguous
    w_ptr,               # *float32, weight (H, L, H) contiguous
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H) contiguous
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_h,    # strides for w (H, L, H)
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

    # For each input channel i in [0..H-1], compute dot with w[:, l, h]
    for i in range(0, H):
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + i * stride_y_h  # (1,128)
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)

        w_ptrs = w_ptr + h_offsets[:, None] * stride_w_o + l_offsets[None, :] * stride_w_l + i * stride_w_h  # (64,128)
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)

        acc += y_vals * w_vals  # broadcast y_vals along h

    # Add bias
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store result to output[b, l, h]
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
        # Ensure float32 and contiguous
        x_t = x.transpose(1, 2).contiguous().to(torch.float32)   # (B, H, L)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)   # (3H, H, L)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)       # (3H)
        conv_weight = conv_weight.contiguous().to(torch.float32)         # (H, H, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)             # (H)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32) # (H, L, H)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)     # (H)

        B, H, L = x_t.shape

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x_t, in_proj_weight, in_proj_bias, BCx,
            B, H, L,
            x_t.stride(0), x_t.stride(1), x_t.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4
        )

        # 2) Chunk BCx (B, L, 3H) -> B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
        BCx_T = BCx.transpose(1, 2).contiguous()   # (B, L, 3H)
        B_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        C_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        x_proj = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_T, B_tensor, C_tensor, x_proj,
            B, L, H,
            BCx_T.stride(0), BCx_T.stride(1), BCx_T.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            num_warps=4
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4
        )

        # 4) Grouped causal 1D conv: Bx -> conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_gate2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate2](
            C_tensor, conv_out, y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4
        )

        # 6) Final out-projection: y -> output (B, L, H)
        output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B, L, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4
        )

        return output


def run(*args):
    return ModelNew()(*args)
