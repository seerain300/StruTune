import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx of shape (B, 3H, L)
# x: (B, L, H), in_proj_weight: (3H, H, L), bias: (3H)
# Output: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *float32, x (B, L, H)
    w_ptr,                 # *float32, in_proj_weight (3H, H, L)
    b_ptr,                 # *float32, in_proj_bias (3H)
    out_ptr,               # *float32, BCx (B, 3H, L)
    B, L, H, J,            # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_out_b, stride_out_j, stride_out_l  # strides for BCx
):
    # Grid: (J_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulator
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Compute output BCx[b, j, l] = sum_k w[j, k, l] * x[b, l, k] + bias[j]
    for k in range(0, H):
        # Load x[b, l, k]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + k * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128) broadcast later

        # Load w[j, k, l]
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + k * stride_w_k + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        # Accumulate
        acc += w_vals * x_vals  # broadcasting x_vals along j dimension

    # Add bias
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk BCx (B, 3H, L) -> (B, L, 3H) -> split into B, C, x_proj via Triton
# inp: (B, L, 3H), outputs: B (B,L,H), C (B,L,H), x (B,L,H)
@triton.jit
def chunk3_kernel(
    inp_ptr,          # *float32, input (B, L, 3H)
    outB_ptr,         # *float32, output B (B, L, H)
    outC_ptr,         # *float32, output C (B, L, H)
    outX_ptr,         # *float32, output x_proj (B, L, H)
    B, L, H,          # sizes
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

    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)

    out_vals = a_vals * b_vals

    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, out_vals, mask=mask)


# 4) Grouped causal 1D convolution: F.conv1d on Bx with kernel_size=4, stride=1, groups=H
# Bx: (B, H, L), conv_weight: (H, H, 4), bias: (H)
# Output: conv_out (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    inp_ptr,            # *float32, input Bx (B, H, L)
    w_ptr,              # *float32, conv_weight (H, H, 4)
    b_bias_ptr,         # *float32, bias (H)
    out_ptr,            # *float32, output (B, H, L)
    B, L, H,            # sizes
    stride_inp_b, stride_inp_h, stride_inp_l,   # strides for inp (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,         # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l    # strides for out (B, H, L)
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
    # For each output channel h, sum over its input channel h and kernel positions
    for ho in range(0, H):
        # inp[b, ho, l]
        in_ptrs = inp_ptr + b * stride_inp_b + ho * stride_inp_h + l_offsets[None, :] * stride_inp_l
        in_vals = tl.load(in_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast later

        # Accumulate conv along kernel positions k in [0, 4)
        for k in range(0, K):
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            acc += w_val * in_vals  # broadcast w_val to (64,128)

    # Add bias
    b_vals = tl.load(b_bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H) and bias (H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_h, stride_w_l, stride_w_k,    # strides for w (H, L, H)
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

    # Compute out[b, h, l] = sum_k w[h, l, k] * y[b, l, k] + b[h]
    for ho in range(0, H):
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)
        # load w[h, l, k] across l and k (k loop implicitly by summing over L dimension)
        # Here we sum over K = L dimension by looping: for each l, multiply with corresponding y_vals
        # But w has HxLxH. We need to treat K dimension as L. To be faithful to original (H, L, H),
        # the operation should be: out = sum over k in L: w[h, k, l] * y[b, l, k] + b[h].
        # That is, for each h, l, sum over k (the L dimension of weight) of w[h, k, l] * y[b, l, k].
        # Implementing this: precompute y_vals (B,L,H) and iterate k over H? Not correct.
        # The original PyTorch uses weight (H, L, H) in F.linear(y, w, b), which implies:
        # out[b, h, l] = sum over k in H: w[h, l, k] * y[b, l, k] + b[h].
        # So weight is (H, L, H). We'll adjust loads accordingly.
        for k in range(0, H):
            w_ptrs = w_ptr + ho * stride_w_h + l_offsets[None, :] * stride_w_l + k * stride_w_k
            w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128)
            acc += w_vals * y_vals  # broadcasting y_vals over ho dimension

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure contiguous and float32
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        B, L, H = x.shape
        J = 3 * H

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, J, L), dtype=torch.float32, device=x.device)

        stride_x_b, stride_x_l, stride_x_h = x.stride()
        stride_w_j, stride_w_k, stride_w_l = in_proj_weight.stride()
        stride_out_b, stride_out_j, stride_out_l = BCx.stride()

        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H, J,
            stride_x_b, stride_x_l, stride_x_h,
            stride_w_j, stride_w_k, stride_w_l,
            stride_out_b, stride_out_j, stride_out_l,
            num_warps=4, num_stages=2
        )

        # 2) Chunk BCx (B, J, L) into B, C, x_proj (B, L, H)
        BCx_T = BCx.view(B, L, J).transpose(-1, -2).contiguous()  # (B, L, 3H)
        B_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        C_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        x_proj = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_inp_b, stride_inp_l, stride_inp_j = BCx_T.stride()  # (B, L, 3H)
        stride_out_b, stride_out_l, stride_out_h = B_tensor.stride()  # (B, L, H)

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_T, B_tensor, C_tensor, x_proj,
            B, L, H,
            stride_inp_b, stride_inp_l, stride_inp_j,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        stride_a_b, stride_a_l, stride_a_h = B_tensor.stride()
        stride_b_b, stride_b_l, stride_b_h = x_proj.stride()
        stride_out_b, stride_out_l, stride_out_h = Bx.stride()

        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            stride_a_b, stride_a_l, stride_a_h,
            stride_b_b, stride_b_l, stride_b_h,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal conv: conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)

        stride_inp_b, stride_inp_h, stride_inp_l = Bx.stride()
        stride_w_o, stride_w_i, stride_w_k = conv_weight.stride()
        stride_out_b, stride_out_h, stride_out_l = conv_out.stride()

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            stride_inp_b, stride_inp_h, stride_inp_l,
            stride_w_o, stride_w_i, stride_w_k,
            stride_out_b, stride_out_h, stride_out_l,
            num_warps=4, num_stages=2
        )

        # 5) Output gating and final projection: y = C_tensor * conv_out, then out_proj
        y = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        stride_y_b, stride_y_l, stride_y_h = C_tensor.stride()
        stride_w_h, stride_w_l, stride_w_k = out_proj_weight.stride()
        stride_out_b, stride_out_l, stride_out_h = y.stride()

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_out](
            C_tensor, conv_out, y,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_h, stride_w_l, stride_w_k,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        output = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        grid_proj = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_proj](
            y, out_proj_weight, out_proj_bias, output,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_h, stride_w_l, stride_w_k,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
