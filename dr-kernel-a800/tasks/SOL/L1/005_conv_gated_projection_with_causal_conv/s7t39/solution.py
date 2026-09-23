import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx of shape (B, 3H, L)
# x: (B, L, H), w: (3H, H, L) => BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                  # *float32, input x (B, L, H) contiguous
    w_ptr,                  # *float32, in_proj_weight (3H, H, L) contiguous
    b_ptr,                  # *float32, in_proj_bias (3H)
    BCx_ptr,                # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,                # sizes
    stride_x_b, stride_x_l, stride_x_h,     # strides for x
    stride_w_j, stride_w_h, stride_w_l,     # strides for w (3H, H, L)
    stride_BC_b, stride_BC_j, stride_BC_l   # strides for BCx (B, 3H, L)
):
    # Grid: (3H_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)    # [64] across 3H
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128] across L

    mask_j = j_offsets < (3 * H)
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulate over H input channels for each j (output channel)
    acc = tl.zeros((64, 128), dtype=tl.float32)

    for ho in range(0, H):
        # Load w[j, ho, l]
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + ho * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        # Load x[b, l, ho]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast along j_offsets

        # Fused multiply-add
        acc += w_vals * x_vals

    # Add bias for j
    b_j = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += b_j[:, None]

    # Store to BCx[b, j, l]
    BC_ptrs = BCx_ptr + b * stride_BC_b + j_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    tl.store(BC_ptrs, acc, mask=mask)


# 2) Split BCx (B, 3H, L) into three parts: B_tensor, C_tensor, x_proj, each (B, L, H)
# Directly load slices in Triton to avoid torch.chunk.
@triton.jit
def chunk3_kernel(
    BCx_ptr, B_tensor_ptr, C_tensor_ptr, x_proj_ptr,
    B, L, H,
    stride_BC_b, stride_BC_j, stride_BC_l,
    stride_B_b, stride_B_l, stride_B_h,
    stride_C_b, stride_C_l, stride_C_h,
    stride_X_b, stride_X_l, stride_X_h
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)    # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    # Load from BCx for each of the three slices
    # B_tensor: j = 0..H-1
    BC_ptrs_B = BCx_ptr + b * stride_BC_b + (h_offsets[:, None]) * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_B = tl.load(BC_ptrs_B, mask=mask, other=0.0)
    B_out_ptrs = B_tensor_ptr + b * stride_B_b + l_offsets[None, :] * stride_B_l + h_offsets[:, None] * stride_B_h
    tl.store(B_out_ptrs, vals_B, mask=mask)

    # C_tensor: j = H..2H-1
    BC_ptrs_C = BCx_ptr + b * stride_BC_b + (H + h_offsets[:, None]) * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_C = tl.load(BC_ptrs_C, mask=mask, other=0.0)
    C_out_ptrs = C_tensor_ptr + b * stride_C_b + l_offsets[None, :] * stride_C_l + h_offsets[:, None] * stride_C_h
    tl.store(C_out_ptrs, vals_C, mask=mask)

    # x_proj: j = 2H..3H-1
    BC_ptrs_X = BCx_ptr + b * stride_BC_b + (2 * H + h_offsets[:, None]) * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_X = tl.load(BC_ptrs_X, mask=mask, other=0.0)
    X_out_ptrs = x_proj_ptr + b * stride_X_b + l_offsets[None, :] * stride_X_l + h_offsets[:, None] * stride_X_h
    tl.store(X_out_ptrs, vals_X, mask=mask)


# 3) Element-wise gating: out = a * b (vectorized), a: B_tensor, b: x_proj
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


# 4) Pad Bx on the last dimension (left) by 3 for causal conv with kernel_size=4
# Bx: (B, H, L) contiguous; out_pad: (B, H, L+3) contiguous
@triton.jit
def pad_left_kernel(
    Bx_ptr,                # *float32, input (B, H, L) contiguous
    out_ptr,               # *float32, output (B, H, Lpad) contiguous
    B, L, H, Lpad,         # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,
    stride_out_b, stride_out_h, stride_out_l
):
    # Grid: (H_tiles, Lpad_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)    # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128], indices in [0, Lpad)

    mask_h = h_offsets < H
    mask_l = l_offsets < Lpad
    mask = mask_h[:, None] & mask_l[None, :]

    # For positions l < L, copy from Bx; else zeros
    l_in = l_offsets - 3
    valid_in = mask & (l_in >= 0) & (l_in < L)
    in_ptrs = Bx_ptr + b * stride_Bx_b + h_offsets[:, None] * stride_Bx_h + l_in[None, :] * stride_Bx_l
    vals = tl.load(in_ptrs, mask=valid_in, other=0.0)  # (64, 128)

    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, vals, mask=mask)


# 5) Grouped causal 1D convolution:
# Input Bx_pad: (B, H, Lpad), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv_group_causal_kernel(
    Bx_pad_ptr,            # *float32, input Bx_pad (B, H, Lpad) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, Lpad, H,         # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx_pad (B, H, Lpad)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l   # strides for out (B, H, L)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)    # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # K = 4
    for k in range(0, 4):
        # For each input channel h, accumulate contributions from its own output channel h and kernel position k
        for ho in range(0, H):
            # Load w[h, h, k]
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar

            # Load Bx_pad[b, h, l_offsets - k]
            t = l_offsets - k
            valid = (t >= 0) & (t < Lpad) & mask
            in_ptrs = Bx_pad_ptr + b * stride_Bx_b + ho * stride_Bx_h + t * stride_Bx_l
            x_vals = tl.load(in_ptrs, mask=valid, other=0.0)  # (64,128)

            # Fused accumulation
            acc += w_val * x_vals

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store conv_out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 6) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H) and bias (H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,    # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h   # strides for out (B, L, H)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)    # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h, accumulate over input channel h and positions l
    for ho in range(0, H):
        # Load w[ho, :, ho], which is a vector over L
        w_ptrs = w_ptr + ho * stride_w_o + l_offsets[None, :] * stride_w_l + ho * stride_w_i  # (1,128)
        w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along ho

        # Load y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)

        acc += w_vals * y_vals  # broadcast along ho (since ho is scalar here)

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store out[b, h, l]
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
        # Shapes:
        Bsz, Lsz, Hsz = x.shape
        assert x.dtype == torch.float32 and in_proj_weight.dtype == torch.float32 and in_proj_bias.dtype == torch.float32
        assert conv_weight.dtype == torch.float32 and conv_bias.dtype == torch.float32
        assert out_proj_weight.dtype == torch.float32 and out_proj_bias.dtype == torch.float32

        # 1) In-projection: BCx (B, 3H, L)
        x_c = x.contiguous()
        w_c = in_proj_weight.contiguous()
        b_c = in_proj_bias.contiguous()
        BCx = torch.empty((Bsz, 3 * Hsz, Lsz), device=x.device, dtype=x.dtype)

        grid_in = (triton.cdiv(3 * Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        in_proj_kernel_B[grid_in](
            x_c, w_c, b_c, BCx,
            Bsz, Lsz, Hsz,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            w_c.stride(0), w_c.stride(1), w_c.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj (each B, L, H) using Triton kernel
        B_tensor = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)
        C_tensor = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)
        x_proj = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)

        grid_chunk = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        chunk3_kernel[grid_chunk](
            BCx, B_tensor, C_tensor, x_proj,
            Bsz, Lsz, Hsz,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)

        grid_gate = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            Bsz, Lsz, Hsz,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Pad Bx on the right for causal conv (kernel_size=4 => pad=3)
        Bx_pad = torch.empty((Bsz, Hsz, Lsz + 3), device=x.device, dtype=x.dtype)

        grid_pad = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz + 3, 128), Bsz)
        pad_left_kernel[grid_pad](
            Bx, Bx_pad,
            Bsz, Lsz, Hsz, Lsz + 3,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Grouped causal conv: conv_out (B, H, L)
        conv_out = torch.empty((Bsz, Hsz, Lsz), device=x.device, dtype=x.dtype)

        grid_conv = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        conv_group_causal_kernel[grid_conv](
            Bx_pad, conv_weight.contiguous(), conv_bias.contiguous(), conv_out,
            Bsz, Lsz, Lsz + 3, Hsz,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 6) Output gating: y = C_tensor * conv_out
        y = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)

        grid_gate2 = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        gate_mul_kernel[grid_gate2](
            C_tensor, conv_out, y,
            Bsz, Lsz, Hsz,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4, num_stages=2
        )

        # 7) Final out-projection: output (B, L, H)
        out = torch.empty((Bsz, Lsz, Hsz), device=x.device, dtype=x.dtype)

        grid_out = (triton.cdiv(Hsz, 64), triton.cdiv(Lsz, 128), Bsz)
        out_proj_kernel[grid_out](
            y, out_proj_weight.contiguous(), out_proj_bias.contiguous(), out,
            Bsz, Lsz, Hsz,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4, num_stages=2
        )

        return out


# For local testing (optional):
# def run_example():
#     B, L, H = 2, 4096, 128
#     x = torch.randn(B, L, H, device='cuda', dtype=torch.float32)
#     in_proj_weight = torch.randn(3*H, H, L, device='cuda', dtype=torch.float32)
#     in_proj_bias = torch.randn(3*H, device='cuda', dtype=torch.float32)
#     conv_weight = torch.randn(H, H, 4, device='cuda', dtype=torch.float32)
#     conv_bias = torch.randn(H, device='cuda', dtype=torch.float32)
#     out_proj_weight = torch.randn(H, L, H, device='cuda', dtype=torch.float32)
#     out_proj_bias = torch.randn(H, device='cuda', dtype=torch.float32)
#     model = ModelNew().cuda()
#     out = model(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)
#     print(out.shape)  # should be (B, L, H)

# run_example()


def run(*args):
    return ModelNew()(*args)
