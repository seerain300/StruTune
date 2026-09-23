import torch
import torch.nn as nn
import triton
import triton.language as tl

# 1) In-projection kernel: compute BCx of shape (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *float32, base pointer to x (B, L, H)
    w_ptr,                 # *float32, base pointer to in_proj_weight (3H, H, L)
    b_ptr,                 # *float32, base pointer to in_proj_bias (3H)
    out_ptr,               # *float32, base pointer to output (B, 3H, L)
    B, L, H,               # int32 sizes: B, L, H
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_out_b, stride_out_j, stride_out_l  # strides for output
):
    # Grid: (J_tiles, L_tiles, B) where J = 3*H
    J = 3 * H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulator for (64, 128)
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Compute BCx[b, j, l] = sum_k in_proj_weight[j, k, l] * x[b, l, k] + bias[j]
    # We iterate over k in [0, H)
    for k in range(0, H):
        # Load x[b, l, k] across l_offsets
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + k * stride_x_h  # shape (1, 128)
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast over j later
        # Load w[j, k, l] across j_offsets and l_offsets
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + k * stride_w_k + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)
        # Multiply and accumulate: broadcast x_vals across j dimension
        acc += w_vals * x_vals  # broadcasting: (64,128) * (1,128) -> (64,128)

    # Add bias
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, j, l]
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk along last dim of (B, 3H, L) into three (B, H, L): B_tensor, C_tensor, x_proj
@triton.jit
def chunk3_kernel(
    inp_ptr,               # *float32, input pointer (B, 3H, L)
    outB_ptr, outC_ptr, outX_ptr,  # *float32, output pointers (B, H, L) for each chunk
    B, L, H,               # int32 sizes
    stride_inp_b, stride_inp_j, stride_inp_l,  # strides for input
    stride_outB_b, stride_outB_j, stride_outB_l,  # strides for output B
    stride_outC_b, stride_outC_j, stride_outC_l,  # strides for output C
    stride_outX_b, stride_outX_j, stride_outX_l   # strides for output x_proj
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    # For B: j in [0, H)
    jB = h_offsets
    inB_ptrs = inp_ptr + b * stride_inp_b + jB[:, None] * stride_inp_j + l_offsets[None, :] * stride_inp_l
    outB_ptrs = outB_ptr + b * stride_outB_b + jB[:, None] * stride_outB_j + l_offsets[None, :] * stride_outB_l
    tl.store(outB_ptrs, tl.load(inB_ptrs, mask=mask, other=0.0), mask=mask)

    # For C: j in [H, 2H)
    jC = h_offsets + H
    inC_ptrs = inp_ptr + b * stride_inp_b + jC[:, None] * stride_inp_j + l_offsets[None, :] * stride_inp_l
    outC_ptrs = outC_ptr + b * stride_outC_b + jC[:, None] * stride_outC_j + l_offsets[None, :] * stride_outC_l
    tl.store(outC_ptrs, tl.load(inC_ptrs, mask=mask, other=0.0), mask=mask)

    # For x_proj: j in [2H, 3H)
    jX = h_offsets + 2 * H
    inX_ptrs = inp_ptr + b * stride_inp_b + jX[:, None] * stride_inp_j + l_offsets[None, :] * stride_inp_l
    outX_ptrs = outX_ptr + b * stride_outX_b + jX[:, None] * stride_outX_j + l_offsets[None, :] * stride_outX_l
    tl.store(outX_ptrs, tl.load(inX_ptrs, mask=mask, other=0.0), mask=mask)


# 3) Elementwise gating: Bx = B_tensor * x_proj
@triton.jit
def gate_mul_kernel(
    a_ptr, b_ptr, out_ptr,  # pointers for A (B, L, H), B (B, L, H), and out (B, L, H)
    B, L, H,
    stride_a_b, stride_a_l, stride_a_h,
    stride_b_b, stride_b_l, stride_b_h,
    stride_out_b, stride_out_l, stride_out_h
):
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)
    l_offsets = l_block * 128 + tl.arange(0, 128)
    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    a_ptrs = a_ptr + b * stride_a_b + l_offsets[None, :] * stride_a_l + h_offsets[:, None] * stride_a_h
    b_ptrs = b_ptr + b * stride_b_b + l_offsets[None, :] * stride_b_l + h_offsets[:, None] * stride_b_h
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h

    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, a_vals * b_vals, mask=mask)


# 4) Grouped causal 1D convolution: conv1d groups=H, kernel_size=4, stride=1, causal padding
@triton.jit
def conv1d_grouped_causal_kernel(
    inp_ptr,            # *float32, input pointer Bx (B, H, L)
    w_ptr,              # *float32, weight pointer (H, H, 4)
    bias_ptr,           # *float32, bias pointer (H)
    out_ptr,            # *float32, output pointer (B, H, L)
    B, L, H,            # int32 sizes
    stride_inp_b, stride_inp_h, stride_inp_l,   # strides for inp (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,         # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l    # strides for out (B, H, L)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    K = 4
    # For each output channel h, sum over its input channel h and kernel positions
    for ho in range(0, H):
        # inp[b, ho, l]
        in_ptrs = inp_ptr + b * stride_inp_b + ho * stride_inp_h + l_offsets[None, :] * stride_inp_l
        in_vals = tl.load(in_ptrs, mask=mask_l[None, :], other=0.0)  # (1,128), broadcast along ho later

        # Accumulate conv along kernel positions k in [0, 4)
        for k in range(0, K):
            # weight w[ho, ho, k] scalar, since groups=H means output channel = input channel
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
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # int32 sizes
    stride_y_b, stride_y_l, stride_y_h,     # strides for y
    stride_w_h, stride_w_l, stride_w_k,     # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h  # strides for out
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # out[b, l, h] = sum_k w[h, l, k] * y[b, l, k] + b[h]
    for k in range(0, H):  # K=L, but out_proj_weight is (H, L, H), we loop over k in H (input dim)
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + k * stride_y_h  # (1,128)
        w_ptrs = w_ptr + h_offsets[:, None] * stride_w_h + l_offsets[None, :] * stride_w_l + k * stride_w_k  # (64,128)
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)
        acc += w_vals * y_vals  # (64,128)

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
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
            B, L, H,
            stride_x_b, stride_x_l, stride_x_h,
            stride_w_j, stride_w_k, stride_w_l,
            stride_out_b, stride_out_j, stride_out_l,
            num_warps=4, num_stages=2
        )

        # 2) Chunk along last dim: (B, 3H, L) -> B_tensor, C_tensor, x_proj (B, H, L)
        B_tensor = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        C_tensor = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        x_proj = torch.empty((B, H, L), dtype=torch.float32, device=x.device)

        stride_inp_b, stride_inp_j, stride_inp_l = BCx.stride()
        stride_outB_b, stride_outB_j, stride_outB_l = B_tensor.stride()
        stride_outC_b, stride_outC_j, stride_outC_l = C_tensor.stride()
        stride_outX_b, stride_outX_j, stride_outX_l = x_proj.stride()

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx, B_tensor, C_tensor, x_proj,
            B, L, H,
            stride_inp_b, stride_inp_j, stride_inp_l,
            stride_outB_b, stride_outB_j, stride_outB_l,
            stride_outC_b, stride_outC_j, stride_outC_l,
            stride_outX_b, stride_outX_j, stride_outX_l,
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj (B, L, H)
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

        # 4) Grouped causal 1D conv: conv_out (B, H, L)
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

        # 5) Output gating: y = C_tensor * conv_out
        y = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_y_b, stride_y_l, stride_y_h = conv_out.stride()
        stride_w_h, stride_w_l, stride_w_k = out_proj_weight.stride()
        stride_out_b, stride_out_l, stride_out_h = y.stride()

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        # Note: original code uses y = C * conv_out; we have y here as output name,
        # but we compute conv_out and then apply C * conv_out.
        # We'll pass conv_out as y_ptr's input and out as y.
        # However, to avoid confusion, we can simply compute y = C_tensor * conv_out using PyTorch multiply here.
        # Since the heavy op was conv, and this is elementwise multiply, it's acceptable and fast.
        y = C_tensor * conv_out

        # 6) Out-projection to final (B, L, H)
        final_out = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_y_b, stride_y_l, stride_y_h = y.stride()
        stride_w_h, stride_w_l, stride_w_k = out_proj_weight.stride()
        stride_out_b, stride_out_l, stride_out_h = final_out.stride()

        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, final_out,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_h, stride_w_l, stride_w_k,
            stride_out_b, stride_out_l, stride_out_h,
            num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
