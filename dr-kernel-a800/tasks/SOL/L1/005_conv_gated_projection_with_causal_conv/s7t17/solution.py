import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: BCx = x @ in_proj_weight^T + in_proj_bias
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                # *float32, input x (B, L, H)
    w_ptr,                # *float32, in_proj_weight (3H, H, L)
    b_ptr,                # *float32, in_proj_bias (3H)
    BCx_ptr,              # *float32, output BCx (B, 3H, L)
    B, L, H,              # sizes
    stride_x_b, stride_x_l, stride_x_h,    # strides for x (B, L, H)
    stride_w_j, stride_w_h, stride_w_l,    # strides for w (3H, H, L)
    stride_BCj, stride_BC_b, stride_BC_l   # strides for BCx (B, 3H, L)
):
    # Grid: (3H_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    J = 3 * H
    j_offsets = j_block * 64 + tl.arange(0, 64)        # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)      # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulator for BCx[b, j, l]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output j, accumulate over input channels h and sequence l
    for ho in range(0, H):  # input channel index varies over H; weight index j varies over 3H
        # Load x[b, l, ho] for all l in block
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast later

        # Load weight w[j, ho, l] for j_offsets and l_offsets
        # w has shape (3H, H, L). For fixed ho (input channel), w[j, ho, l] varies with j and l.
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + ho * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        # Accumulate: BCx[b, j, l] += x[b, l, ho] * w[j, ho, l]
        acc += x_vals * w_vals  # broadcast x_vals along j dimension

    # Add bias
    bias_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to BCx[b, j, l]
    BCx_ptrs = BCx_ptr + b * stride_BC_b + j_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    tl.store(BCx_ptrs, acc, mask=mask)


# 2) Chunk 3: split BCx (B, 3H, L) into three (B, L, H): B_tensor, C_tensor, x_proj
@triton.jit
def chunk3_kernel(
    BCx_ptr,               # *float32, input BCx (B, 3H, L)
    B_out_ptr,             # *float32, output B_tensor (B, L, H)
    C_out_ptr,             # *float32, output C_tensor (B, L, H)
    xproj_out_ptr,         # *float32, output x_proj (B, L, H)
    B, L, H,               # sizes
    stride_BC_b, stride_BC_j, stride_BC_l,   # strides for BCx (B, 3H, L)
    stride_B_b, stride_B_l, stride_B_h,      # strides for B_out (B, L, H)
    stride_C_b, stride_C_l, stride_C_h,      # strides for C_out (B, L, H)
    stride_x_b, stride_x_l, stride_x_h       # strides for xproj_out (B, L, H)
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

    # Copy x_proj: BCx[b, l, 0..H-1]
    for ho in range(0, H):
        src_ptrs = BCx_ptr + b * stride_BC_b + ho * stride_BC_j + l_offsets[None, :] * stride_BC_l
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        dst_ptrs_x = xproj_out_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        tl.store(dst_ptrs_x, vals, mask=mask)

    # Copy B: BCx[b, l, H..2H-1]
    for ho in range(0, H):
        src_ptrs = BCx_ptr + b * stride_BC_b + (H + ho) * stride_BC_j + l_offsets[None, :] * stride_BC_l
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        dst_ptrs_B = B_out_ptr + b * stride_B_b + l_offsets[None, :] * stride_B_l + ho * stride_B_h
        tl.store(dst_ptrs_B, vals, mask=mask)

    # Copy C: BCx[b, l, 2H..3H-1]
    for ho in range(0, H):
        src_ptrs = BCx_ptr + b * stride_BC_b + (2 * H + ho) * stride_BC_j + l_offsets[None, :] * stride_BC_l
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        dst_ptrs_C = C_out_ptr + b * stride_C_b + l_offsets[None, :] * stride_C_l + ho * stride_C_h
        tl.store(dst_ptrs_C, vals, mask=mask)


# 3) Element-wise gating: out = a * b
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
# Causal padding handled by masked loads for kernel positions k in [0,4).
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_o, stride_w_h, stride_w_k,      # strides for w (H, H, 4)
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

    # Sum over K=4 causal taps (K-1 padding handled by masked loads)
    for k in range(0, 4):
        # Input index for this tap: l_k = l + (K-1 - k)
        # For k=0: l_k = l + 3; k=1: l+2; k=2: l+1; k=3: l (causal)
        # Masked load: if out of bounds, set to 0
        l_k = l_offsets[None, :] + (3 - k)
        in_bounds = l_k < L

        # y[b, l_k, h]
        y_ptrs = Bx_ptr + b * stride_Bx_b + h_offsets[:, None] * stride_Bx_h + l_k * stride_Bx_l
        y_vals = tl.load(y_ptrs, mask=mask & in_bounds, other=0.0)  # (64,128)

        # w[h, h, k] scalar per h
        w_ptrs = w_ptr + h_offsets * stride_w_o + h_offsets * stride_w_h + k * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_h, other=0.0)  # (64,)
        w_vals = w_vals[:, None]  # broadcast along l

        acc += y_vals * w_vals

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H) and bias (H)
@triton.jit
def out_proj_kernel(
    y_ptr,                 # *float32, input y (B, L, H)
    w_ptr,                 # *float32, weight (H, L, H)
    b_ptr,                 # *float32, bias (H)
    out_ptr,               # *float32, output (B, L, H)
    B, L, H,               # sizes
    stride_y_b, stride_y_l, stride_y_h,     # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,     # strides for w (H, L, H)
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

    # For each output channel h, compute sum over input channels i (also H) of w[h, l, i] * y[b, l, i]
    for hi in range(0, H):
        # y[b, l, hi]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hi * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)

        # w[hi, l, hi] vector over l
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
        # Ensure all inputs are CUDA and contiguous; use float32
        assert x.is_cuda, "x must be CUDA tensor"
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All parameters must be CUDA tensors"

        B, L, H = x.shape
        device = x.device
        dtype = torch.float32

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=device, dtype=dtype)
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x.contiguous(),
            in_proj_weight.contiguous(),
            in_proj_bias.contiguous(),
            BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Split BCx into three (B, L, H): B_tensor, C_tensor, x_proj
        B_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        C_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        x_proj = torch.empty((B, L, H), device=device, dtype=dtype)
        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx,
            B_tensor,
            C_tensor,
            x_proj,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            num_warps=4, num_stages=2,
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj (B, L, H)
        Bx = torch.empty((B, L, H), device=device, dtype=dtype)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor,
            x_proj,
            Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2,
        )

        # 4) Grouped causal 1D convolution: Bx (B, H, L), conv_weight (H, H, 4), conv_bias (H) -> conv_out (B, H, L)
        Bx_trans = Bx.transpose(1, 2).contiguous()  # (B, L, H)
        conv_out = torch.empty((B, H, L), device=device, dtype=dtype)  # output (B, H, L)
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_trans,                   # (B, L, H)
            conv_weight.contiguous(),  # (H, H, 4)
            conv_bias.contiguous(),    # (H,)
            conv_out,                  # (B, H, L)
            B, L, H,
            Bx_trans.stride(0), Bx_trans.stride(1), Bx_trans.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2,
        )

        # 5) Final out-projection: y = conv_out.transpose(1,2) -> (B, L, H)
        y = conv_out.transpose(1, 2).contiguous()  # (B, L, H)
        output = torch.empty((B, L, H), device=device, dtype=dtype)
        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y,
            out_proj_weight.contiguous(),  # (H, L, H)
            out_proj_bias.contiguous(),    # (H,)
            output,
            B, L, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2,
        )

        return output


# For completeness, original Model's run is not used in evaluation.
# The entry point class ModelNew is expected to be present and callable.


def run(*args):
    return ModelNew()(*args)
