import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx = x @ in_proj_weight^T + in_proj_bias, shape (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                   # *float32, input x (B, L, H), contiguous
    w_ptr,                   # *float32, weight (3H, H, L), contiguous
    bias_ptr,                # *float32 or 0, bias (3H)
    BCx_ptr,                 # *float32, output (B, 3H, L), contiguous
    B, L, H, threeH,         # sizes
    stride_x_b, stride_x_l, stride_x_h,    # strides for x (B, L, H)
    stride_w_t, stride_w_h, stride_w_l,    # strides for w (3H, H, L)
    stride_BCx_b, stride_BCx_t, stride_BCx_l   # strides for BCx (B, 3H, L)
):
    # 3D grid: (tiles over 3H, tiles over L, B)
    t_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    t_offsets = t_block * 64 + tl.arange(0, 64)    # [64] over 3H dimension
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128] over L dimension

    mask_t = t_offsets < threeH
    mask_l = l_offsets < L
    mask = mask_t[:, None] & mask_l[None, :]

    # Accumulator for current tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over H to compute BCx[b, t, l] for all t in 3H
    # For each t, t corresponds to one input channel ho and one output feature j.
    # x[b, l, ho] * w[t, ho, l] where t = j*H + ho
    for ho in range(0, H):
        # x[b, l, ho]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h  # (1, 128)
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along t
        # w[t, ho, l]
        w_ptrs = w_ptr + t_offsets[:, None] * stride_w_t + ho * stride_w_h + l_offsets[None, :] * stride_w_l  # (64,128)
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)
        # accumulate
        acc += x_vals * w_vals  # broadcasting x_vals across t dimension

    # Add bias if provided
    if bias_ptr != 0:
        b_vals = tl.load(bias_ptr + t_offsets, mask=mask_t, other=0.0)  # (64,)
        acc += b_vals[:, None]

    # Store to BCx[b, t, l]
    BCx_ptrs = BCx_ptr + b * stride_BCx_b + t_offsets[:, None] * stride_BCx_t + l_offsets[None, :] * stride_BCx_l
    tl.store(BCx_ptrs, acc, mask=mask)


# 2) Chunking: split (B, 3H, L) into three (B, L, H): B_tensor, C_tensor, x_proj
# 3D grid: (H_tiles, L_tiles, B) to cover all channels
@triton.jit
def chunk3_kernel(
    inp_ptr,               # *float32, input BCx (B, 3H, L), contiguous
    outB_ptr, outC_ptr, outX_ptr,  # *float32, outputs (B, L, H), contiguous
    B, L, H,               # sizes
    stride_inp_b, stride_inp_t, stride_inp_l,   # strides for inp (B, 3H, L)
    stride_out_b, stride_out_l, stride_out_h    # strides for outputs (B, L, H)
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
        in_ptrs_x = inp_ptr + b * stride_inp_b + ho * stride_inp_t + l_offsets[None, :] * stride_inp_l
        vals_x = tl.load(in_ptrs_x, mask=mask, other=0.0)  # (64,128)
        out_ptrs_x = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_x, vals_x, mask=mask)

    # Copy B from inp[b, l, H..2H-1]
    for ho in range(0, H):
        in_ptrs_B = inp_ptr + b * stride_inp_b + (H + ho) * stride_inp_t + l_offsets[None, :] * stride_inp_l
        vals_B = tl.load(in_ptrs_B, mask=mask, other=0.0)  # (64,128)
        out_ptrs_B = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, vals_B, mask=mask)

    # Copy C from inp[b, l, 2H..3H-1]
    for ho in range(0, H):
        in_ptrs_C = inp_ptr + b * stride_inp_b + (2 * H + ho) * stride_inp_t + l_offsets[None, :] * stride_inp_l
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
# Output conv_out: (B, H, L)
# We assume Bx is padded on the last dimension by 3 zeros (causal).
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L), contiguous
    w_ptr,                 # *float32, weight (H, H, 4), contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L), contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_h, stride_w_h2, stride_w_k,     # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l # strides for out (B, H, L)
):
    # 3D grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Kernel size K = 4, causal padding handled by host: we assume Bx_ptr already includes 3 leading zeros.
    for k in range(0, 4):
        # Sum over input channel equals output channel h (groups=H)
        bx_ptrs_k = Bx_ptr + b * stride_Bx_b + h_offsets[:, None] * stride_Bx_h + (l_offsets[None, :] + k) * stride_Bx_l  # (64,128)
        bx_vals_k = tl.load(bx_ptrs_k, mask=mask, other=0.0)  # (64,128)
        # Weight w[h, h, k] is a vector of length H at this fixed k
        w_ptrs_k = w_ptr + h_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_h2 + k * stride_w_k  # (64,64)
        w_vals_k = tl.load(w_ptrs_k, mask=mask_h[:, None], other=0.0)  # (64,64)
        # Multiply and reduce along h dimension (broadcast w_vals_k along rows)
        # Note: broadcasting (64,128) * (64,64) requires same h_offsets; we align h dimension explicitly:
        acc += tl.sum(bx_vals_k * w_vals_k, axis=1)[:, None]

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,   # strides for y (B, L, H)
    stride_w_t, stride_w_l, stride_w_h,   # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h   # strides for out (B, L, H)
):
    # 3D grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    # Accumulator for current tile (B, H, L_tile)
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output h, sum over input h and L
    for ho in range(0, H):
        # y[b, l, ho] -> (1, 128), broadcast across h_offsets
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h  # (1,128)
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)
        # w[ho, l, ho] -> (128,)
        w_ptrs = w_ptr + ho * stride_w_t + l_offsets[None, :] * stride_w_l + ho * stride_w_h  # (1,128)
        w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)
        # broadcast multiply
        acc += y_vals * w_vals  # (64,128)

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    """
    Triton-optimized fused pipeline:
    1) In-projection: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, 3H, L)
    2) Split into B_tensor, C_tensor, x_proj (each B, L, H)
    3) Element-wise gating: Bx = B_tensor * x_proj
    4) Grouped causal 1D convolution: conv_out (B, H, L)
    5) Output gating: y = C_tensor * conv_out
    6) Final out-projection: output (B, L, H)
    """
    # Ensure float32 and contiguous
    x = x.contiguous().to(torch.float32)
    in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
    in_proj_bias = in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else None
    conv_weight = conv_weight.contiguous().to(torch.float32)
    conv_bias = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else None
    out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
    out_proj_bias = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else None

    B, L, H = x.shape
    threeH = 3 * H

    # 1) In-projection
    BCx = torch.empty((B, threeH, L), device=x.device, dtype=torch.float32)
    grid_in = (triton.cdiv(threeH, 64), triton.cdiv(L, 128), B)
    in_proj_kernel_B[grid_in](
        x, in_proj_weight, in_proj_bias if in_proj_bias is not None else 0, BCx,
        B, L, H, threeH,
        x.stride(0), x.stride(1), x.stride(2),
        in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
        BCx.stride(0), BCx.stride(1), BCx.stride(2)
    )

    # 2) Chunking: split BCx (B, 3H, L) into B_tensor, C_tensor, x_proj (each B, L, H)
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
        x_proj.stride(0), x_proj.stride(1), x_proj.stride(2)
    )

    # 3) Element-wise gating: Bx = B_tensor * x_proj
    Bx = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
    grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
    gate_mul_kernel[grid_gate](
        B_tensor, x_proj, Bx,
        B, L, H,
        B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
        x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
        Bx.stride(0), Bx.stride(1), Bx.stride(2)
    )

    # 4) Grouped causal 1D convolution: conv_out (B, H, L)
    # Create Bx_padded by padding last dimension with 3 zeros on the host to simulate causal conv.
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad left by 3, right by 0
    conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
    grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
    conv1d_grouped_causal_kernel[grid_conv](
        Bx_padded, conv_weight, conv_bias if conv_bias is not None else 0, conv_out,
        B, L, H,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)
    )

    # 5) Output gating: y = C_tensor * conv_out
    y = C_tensor * conv_out  # (B, L, H), elementwise

    # 6) Final out-projection: F.linear(y, out_proj_weight, out_proj_bias) -> (B, L, H)
    output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
    grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
    out_proj_kernel[grid_out](
        y, out_proj_weight, out_proj_bias if out_proj_bias is not None else 0, output,
        B, L, H,
        y.stride(0), y.stride(1), y.stride(2),
        out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
        output.stride(0), output.stride(1), output.stride(2)
    )

    return output


class ModelNew(nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
