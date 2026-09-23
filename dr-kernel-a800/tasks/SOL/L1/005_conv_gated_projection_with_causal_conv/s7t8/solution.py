import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx of shape (B, 3H, L)
# Input x: (B, L, H), in_proj_weight: (3H, H, L), bias: (3H)
# Output BCx_out: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *float32, x (B, L, H)
    w_ptr,                 # *float32, in_proj_weight (3H, H, L)
    b_ptr,                 # *float32, in_proj_bias (3H)
    BCx_ptr,               # *float32, output (B, 3H, L)
    B, L, H, J,            # sizes: B=batch, L=seq_len, H=hidden, J=3*H
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_BCx_b, stride_BCx_j, stride_BCx_l   # strides for output (B, J, L)
):
    # Grid: (J_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_j = j_offsets < J
    mask_l = l_offsets < L

    # For each j in tile, compute BCx[b, j, l] = sum_h w[j, h, l] * x[b, l, h] + bias[j]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over h = 0..H-1
    for h in range(0, H):
        # Load x[b, l, h] across l_offsets
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + h * stride_x_h  # (1,128) but broadcast over j tile
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast later

        # Load w[j, h, l] for all j in tile
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + h * stride_w_k + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=(mask_j[:, None] & mask_l[None, :]), other=0.0)  # (64,128)

        acc += w_vals * x_vals  # broadcast x_vals over j tile

    # Add bias
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to BCx[b, j, l]
    BCx_ptrs = BCx_ptr + b * stride_BCx_b + j_offsets[:, None] * stride_BCx_j + l_offsets[None, :] * stride_BCx_l
    tl.store(BCx_ptrs, acc, mask=(mask_j[:, None] & mask_l[None, :]))


# 2) Chunk along feature dimension: split (B, L, 3H) into three (B, L, H)
# Input inp: (B, L, 3H) contiguous; outputs B, C, x_proj each (B, L, H)
@triton.jit
def chunk3_kernel(
    inp_ptr, outB_ptr, outC_ptr, outX_ptr,
    B, L, H,
    stride_inp_b, stride_inp_l, stride_inp_j,    # strides for inp (B, L, 3H)
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
        in_ptrs_x = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + ho * stride_inp_j
        vals_x = tl.load(in_ptrs_x, mask=mask, other=0.0)  # (64,128)
        out_ptrs_x = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_x, vals_x, mask=mask)

    # Copy B from inp[b, l, H..2H-1]
    for ho in range(0, H):
        in_ptrs_B = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + (H + ho) * stride_inp_j
        vals_B = tl.load(in_ptrs_B, mask=mask, other=0.0)  # (64,128)
        out_ptrs_B = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, vals_B, mask=mask)

    # Copy C from inp[b, l, 2H..3H-1]
    for ho in range(0, H):
        in_ptrs_C = inp_ptr + b * stride_inp_b + l_offsets[None, :] * stride_inp_l + (2 * H + ho) * stride_inp_j
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

    K = 4
    # For each output channel ho, sum over its input channel ho and kernel positions
    for ho in range(0, H):
        # Bx[b, ho, l]
        in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_offsets[None, :] * stride_Bx_l
        in_vals = tl.load(in_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast over ho later

        # Accumulate conv along kernel positions k in [0, 4)
        for k in range(0, K):
            # weight w[ho, ho, k] scalar
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            acc += w_val * in_vals  # broadcast w_val to (64,128)

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H)
# Original weight is (H, L, H). We treat it as (H_in=H, L_in=L, H_out=H).
@triton.jit
def out_proj_kernel(
    y_ptr,                 # *float32, input y (B, L, H)
    w_ptr,                 # *float32, weight (H, L, H)
    b_ptr,                 # *float32, bias (H)
    out_ptr,               # *float32, output (B, L, H)
    B, L, H,               # sizes
    stride_y_b, stride_y_l, stride_y_h,       # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,       # strides for w (H, L, H)
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

    # Compute out[b, l, h] = sum_h w[h, l, h] * y[b, l, h] + bias[h]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    for ho in range(0, H):
        # Load y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast later

        # Load w[ho, l, ho]
        w_ptrs = w_ptr + ho * stride_w_o + l_offsets[None, :] * stride_w_l + ho * stride_w_i
        w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along h tile

        acc += w_vals * y_vals  # broadcast along h tile

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure contiguous and float32 for Triton
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        B, L, H = x.shape
        J = 3 * H

        # Step 1: In-projection -> BCx (B, 3H, L)
        BCx = torch.empty((B, J, L), dtype=torch.float32, device=x.device)

        stride_x_b, stride_x_l, stride_x_h = x.stride()
        stride_w_j, stride_w_k, stride_w_l = in_proj_weight.stride()
        stride_BCx_b, stride_BCx_j, stride_BCx_l = BCx.stride()

        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H, J,
            stride_x_b, stride_x_l, stride_x_h,
            stride_w_j, stride_w_k, stride_w_l,
            stride_BCx_b, stride_BCx_j, stride_BCx_l,
            num_warps=4, num_stages=2
        )

        # Reshape to (B, 3H, L) and transpose to (B, L, 3H) for chunking
        BCx_T = BCx.view(B, J, L).transpose(-1, -2).contiguous()  # (B, L, 3H)

        # Step 2: Chunk (B, L, 3H) into B_tensor, C_tensor, x_proj, each (B, L, H)
        B_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        C_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        x_proj = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_BCxT_b, stride_BCxT_l, stride_BCxT_j = BCx_T.stride()
        stride_out_b, stride_out_l, stride_out_h = B_tensor.stride()

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_T, B_tensor, C_tensor, x_proj,
            B, L, H,
            stride_BCxT_b, stride_BCxT_l, stride_BCxT_j,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        # Step 3: Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # Step 4: Grouped causal conv with kernel_size=4, groups=H
        # conv_weight: (H, H, 4); conv_bias: (H)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)

        stride_Bx_b, stride_Bx_h, stride_Bx_l = Bx.stride()
        stride_w_o, stride_w_i, stride_w_k = conv_weight.stride()
        stride_out_b, stride_out_h, stride_out_l = conv_out.stride()

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            stride_Bx_b, stride_Bx_h, stride_Bx_l,
            stride_w_o, stride_w_i, stride_w_k,
            stride_out_b, stride_out_h, stride_out_l,
            num_warps=4, num_stages=2
        )

        # Step 5: Output gating: y = C_tensor * conv_out
        y = C_tensor * conv_out  # (B, H, L)

        # Step 6: Out-projection: y (B, L, H) from y.transpose(-1, -2).contiguous() (B, H, L)
        y_T = y.transpose(-1, -2).contiguous()  # (B, L, H)

        out = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_y_b, stride_y_l, stride_y_h = y_T.stride()
        stride_w_o_w, stride_w_l_w, stride_w_i_w = out_proj_weight.stride()  # (H, L, H)
        stride_out_b_out, stride_out_l_out, stride_out_h_out = out.stride()

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y_T, out_proj_weight, out_proj_bias, out,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_o_w, stride_w_l_w, stride_w_i_w,
            stride_out_b_out, stride_out_l_out, stride_out_h_out,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
