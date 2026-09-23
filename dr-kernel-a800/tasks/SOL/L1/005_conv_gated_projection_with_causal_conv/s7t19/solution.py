import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx = F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                # *float32, input x (B, L, H) contiguous
    w_ptr,                # *float32, weight (3H, H, L) contiguous
    b_ptr,                # *float32, bias (3H)
    BCx_ptr,              # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,              # sizes
    stride_x_b, stride_x_l, stride_x_h,        # strides for x
    stride_w_j, stride_w_h, stride_w_l,        # strides for w
    stride_BC_b, stride_BC_j, stride_BC_l      # strides for BCx
):
    # Grid: (3H_tiles, L_tiles, B)
    J = 3 * H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)      # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)    # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Accumulate over input channels h and sequence l
    # BCx[b, j, l] = sum_h sum_l x[b, l, h] * w[j, h, l] + bias[j]
    for ho in range(0, H):
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast along j

        # For a given output j, weight index is (j, h, l)
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + ho * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        acc += x_vals * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store BCx[b, j, l]
    BCx_ptrs = BCx_ptr + b * stride_BC_b + j_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    tl.store(BCx_ptrs, acc, mask=mask)


# 2) Split BCx (B, 3H, L) into three tensors (B, L, H): B_tensor, C_tensor, x_proj
# Implemented via Triton to avoid torch.chunk.
@triton.jit
def chunk3_kernel(
    BCx_ptr,            # *float32, input BCx (B, 3H, L) contiguous
    B_ptr,              # *float32, output B_tensor (B, L, H) contiguous
    C_ptr,              # *float32, output C_tensor (B, L, H) contiguous
    X_ptr,              # *float32, output x_proj (B, L, H) contiguous
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    stride_BC_b, stride_BC_j, stride_BC_l,
    stride_B_b, stride_B_l, stride_B_h,
    stride_C_b, stride_C_l, stride_C_h,
    stride_X_b, stride_X_l, stride_X_h,
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

    # Copy B_tensor: BCx[:, h, :]
    BCx_ptrs_B = BCx_ptr + b * stride_BC_b + h_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    B_ptrs = B_ptr + b * stride_B_b + l_offsets[None, :] * stride_B_l + h_offsets[:, None] * stride_B_h
    tl.store(B_ptrs, tl.load(BCx_ptrs_B, mask=mask, other=0.0), mask=mask)

    # Copy C_tensor: BCx[:, H + h, :]
    BCx_ptrs_C = BCx_ptr + b * stride_BC_b + (H + h_offsets)[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    C_ptrs = C_ptr + b * stride_C_b + l_offsets[None, :] * stride_C_l + h_offsets[:, None] * stride_C_h
    tl.store(C_ptrs, tl.load(BCx_ptrs_C, mask=mask, other=0.0), mask=mask)

    # Copy x_proj: BCx[:, 2H + h, :]
    BCx_ptrs_X = BCx_ptr + b * stride_BC_b + (2 * H + h_offsets)[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    X_ptrs = X_ptr + b * stride_X_b + l_offsets[None, :] * stride_X_l + h_offsets[:, None] * stride_X_h
    tl.store(X_ptrs, tl.load(BCx_ptrs_X, mask=mask, other=0.0), mask=mask)


# 3) Element-wise gating: out = a * b (vectorized)
# a: (B, L, H), b: (B, L, H), out: (B, L, H)
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
# Input Bx: (B, H, L) -> we pad on sequence by kernel_size-1 (3) for causal
# conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
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

    K = 4
    # For each output channel h, accumulate: out[b, l, h] = sum_i w[h, h, i] * y[b, l+i, h] + bias[h]
    for ho in range(0, H):
        # y[b, l+i, ho] for i in [0..3]
        # Note: causal padding means for i > 0, we read indices l - i where l - i < 0 -> 0
        for i in range(0, K):
            idx = l_offsets[None, :] + i  # vector over l
            # Apply causal padding by masking idx >= 0
            valid = idx >= 0
            # Load y with mask combining valid and l mask
            y_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + idx * stride_Bx_l
            y_vals = tl.load(y_ptrs, mask=mask_l[None, :] & valid, other=0.0)  # (64,128)

            # Load weight w[ho, ho, i] scalar
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + i * stride_w_k
            w_val = tl.load(w_ptrs)

            acc += w_val * y_vals

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H)
# Note: Original code uses weight (H, L, H) for F.linear(y, out_proj_weight, bias)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,
    stride_y_b, stride_y_l, stride_y_h,
    stride_w_o, stride_w_l, stride_w_i,
    stride_out_b, stride_out_l, stride_out_h
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

    # Compute out[b, l, h] = sum_j w[h, l, j] * y[b, l, j] + b[h]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    for ho in range(0, H):
        # y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along h

        # w[ho, l, ho] vector over l
        w_ptrs = w_ptr + ho * stride_w_o + l_offsets[None, :] * stride_w_l + ho * stride_w_i
        w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)

        acc += y_vals * w_vals  # broadcast y_vals along h dimension

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
        # Ensure contiguity and dtype
        device = x.device
        dtype = torch.float32
        x = x.contiguous().to(dtype)
        in_proj_weight = in_proj_weight.contiguous().to(dtype)
        in_proj_bias = in_proj_bias.contiguous().to(dtype)
        conv_weight = conv_weight.contiguous().to(dtype)
        conv_bias = conv_bias.contiguous().to(dtype)
        out_proj_weight = out_proj_weight.contiguous().to(dtype)
        out_proj_bias = out_proj_bias.contiguous().to(dtype)

        B, L, H = x.shape

        # 1) In-projection: BCx of shape (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=device, dtype=dtype)
        J = 3 * H

        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj (B, L, H)
        B_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        C_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        x_proj = torch.empty((B, L, H), device=device, dtype=dtype)

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
        Bx = torch.empty((B, L, H), device=device, dtype=dtype)

        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
        )

        # 4) Grouped causal 1D convolution on Bx with kernel_size=4, stride=1, groups=H
        # We treat Bx as (B, H, L); causal padding means we ignore indices < 0 when reading y[b, l-k, h].
        conv_out = torch.empty((B, H, L), device=device, dtype=dtype)

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        )

        # 5) Output gating and final projection
        # Output gating: y = C_tensor * conv_out, shape (B, L, H)
        y = torch.empty((B, L, H), device=device, dtype=dtype)

        grid_gate2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate2](
            C_tensor, conv_out, y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
        )

        # Final out-projection: out = F.linear(y, out_proj_weight, out_proj_bias)
        out = torch.empty((B, L, H), device=device, dtype=dtype)

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, out,
            B, L, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
        )

        return out


def run(*args):
    return ModelNew()(*args)
