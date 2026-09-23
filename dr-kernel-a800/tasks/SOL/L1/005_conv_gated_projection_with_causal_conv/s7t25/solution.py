import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx of shape (B, 3H, L)
# inp: (B, 3H, L), weight: (3H, H, L), bias: (3H)
@triton.jit
def in_proj_kernel_B(
    inp_ptr,            # *float32, input x (B, L, H) contiguous
    w_ptr,              # *float32, weight (3H, H, L) contiguous
    b_ptr,              # *float32, bias (3H)
    out_ptr,            # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,            # sizes
    stride_inp_b, stride_inp_l, stride_inp_h,      # strides for x (B, L, H)
    stride_w_o, stride_w_i, stride_w_l,            # strides for weight (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l       # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, L_tiles, B) where J = 3H
    j_block = tl.program_id(0)  # over 3H
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    J = 3 * H
    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel j in [0, 3H), sum over input channels i and sequence positions l
    for j in range(0, J):
        # If j >= 3H, mask_j prevents loads/stores, but we guard anyway
        if j >= (3 * H):
            break
        # load weight scalar w[j, i, l] across i and l. We will compute i dynamically via j and weight strides
        # However, Triton requires static loops; instead, we vectorize over i dimension by using channel indexing.
        # Better approach: load x for all i and accumulate over i for each j.
        # Since weight is (3H, H, L), we need to iterate over i in H and l in L.
        # We'll precompute contributions for a fixed j: sum over i in [0..H-1], l in [0..L-1]
        # But to avoid dynamic loops, we'll compute outer products per i: w[j, i, :] * x[b, i, :].
        # We'll do this by iterating i and using x_ptr indexing.
        # Note: Triton supports while loops with runtime condition.
        i = 0
        while i < H:
            # load x[b, i, l] vector
            x_ptrs = inp_ptr + b * stride_inp_b + i * stride_inp_h + l_offsets[None, :] * stride_inp_l  # (1,128), broadcast later
            x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128)
            # load w[j, i, l] vector
            w_ptrs = w_ptr + j * stride_w_o + i * stride_w_i + l_offsets[None, :] * stride_w_l
            w_vals = tl.load(w_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128)
            acc += w_vals * x_vals  # broadcast w_vals along 64 rows
            i += 1

        # add bias if j within 3H
        bias_val = tl.load(b_ptr + j, mask=mask_j, other=0.0)  # scalar
        acc += bias_val

    # Store to out[b, j, l]
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk BCx (B, L, 3H) -> (B, L, H) into three tensors: B, C, x_proj
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
    stride_w_o, stride_w_i, stride_w_k,      # strides for weight (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l
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

    K = 4  # kernel_size
    for ho in range(0, H):
        # base pointer for this output channel
        in_base = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h
        # sum over kernel positions
        for k in range(0, K):
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            in_ptrs = in_base + (l_offsets + k) * stride_Bx_l  # causal shift by k
            in_vals = tl.load(in_ptrs, mask=mask, other=0.0)  # (1, 128)
            acc += w_val * in_vals  # broadcast along rows

    # add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, H)
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

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h, compute dot over input channels i and sequence l
    for ho in range(0, H):
        # y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast later

        # weight w[ho, i, ho] across i
        for i in range(0, H):
            w_ptrs = w_ptr + ho * stride_w_o + i * stride_w_i + ho * stride_w_l
            w_vals = tl.load(w_ptrs)  # scalar
            acc += w_vals * y_vals  # broadcast across rows

    # add bias
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # store
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-optimized fused pipeline:
        1) In-projection: x -> (B, 3H, L)
        2) Transpose (B, 3H, L) -> (B, L, 3H) and chunk into (B, C, x_proj)
        3) Element-wise gating: Bx = B * x_proj
        4) Grouped causal 1D convolution: groups=H, kernel_size=4
        5) Output gating: y = C * conv_out
        6) Final out-projection: y -> (B, L, H)
        """
        # Ensure float32 and contiguous
        device = x.device
        B, L, H = x.shape
        x_c = x.contiguous().to(torch.float32)

        # 1) In-projection
        # BCx: (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=device, dtype=torch.float32)
        in_proj_weight_c = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias_c = in_proj_bias.contiguous().to(torch.float32)

        # strides for x (B, L, H)
        stride_inp_b, stride_inp_l, stride_inp_h = x_c.stride()
        # strides for weight (3H, H, L)
        stride_w_o, stride_w_i, stride_w_l = in_proj_weight_c.stride()
        # strides for BCx (B, 3H, L)
        stride_out_b, stride_out_j, stride_out_l = BCx.stride()

        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x_c, in_proj_weight_c, in_proj_bias_c, BCx,
            B, L, H,
            stride_inp_b, stride_inp_l, stride_inp_h,
            stride_w_o, stride_w_i, stride_w_l,
            stride_out_b, stride_out_j, stride_out_l
        )

        # 2) Transpose and chunk (B, 3H, L) -> (B, L, 3H)
        BCx_T = BCx.transpose(1, 2)  # (B, L, 3H), contiguous
        B_tensor = torch.empty((B, L, H), device=device, dtype=torch.float32)
        C_tensor = torch.empty((B, L, H), device=device, dtype=torch.float32)
        x_proj = torch.empty((B, L, H), device=device, dtype=torch.float32)

        stride_inp_b_T, stride_inp_l, stride_inp_j = BCx_T.stride()  # (B, L, 3H)
        stride_out_b, stride_out_l, stride_out_h = (B_tensor.stride(), C_tensor.stride(), x_proj.stride())
        # Note: All three outputs have identical stride pattern; we pass each separately.

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_T, B_tensor, C_tensor, x_proj,
            B, L, H,
            stride_inp_b_T, stride_inp_l, stride_inp_j,
            stride_out_b, stride_out_l, stride_out_h  # apply to all three tensors uniformly
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, L, H), device=device, dtype=torch.float32)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            *B_tensor.stride(), *x_proj.stride(), *Bx.stride()
        )

        # 4) Grouped causal 1D convolution on Bx: input (B, H, L)
        # Pad for causal conv: conv_weight has kernel_size=4, so pad L by 3 on left
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad last dim by 3 zeros on left
        Bx_padded = Bx_padded.contiguous()

        conv_out = torch.empty((B, H, L), device=device, dtype=torch.float32)
        conv_weight_c = conv_weight.contiguous().to(torch.float32)
        conv_bias_c = conv_bias.contiguous().to(torch.float32)

        stride_Bx_b, stride_Bx_h, stride_Bx_l = Bx_padded.stride()
        stride_w_o, stride_w_i, stride_w_k = conv_weight_c.stride()
        stride_out_b, stride_out_h, stride_out_l = conv_out.stride()

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_padded, conv_weight_c, conv_bias_c, conv_out,
            B, L, H,
            stride_Bx_b, stride_Bx_h, stride_Bx_l,
            stride_w_o, stride_w_i, stride_w_k,
            stride_out_b, stride_out_h, stride_out_l
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, L, H), device=device, dtype=torch.float32)
        grid_gate2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate2](
            C_tensor, conv_out, y,
            B, L, H,
            *C_tensor.stride(), *conv_out.stride(), *y.stride()
        )

        # 6) Out-projection: y -> (B, L, H)
        out = torch.empty((B, L, H), device=device, dtype=torch.float32)
        out_proj_weight_c = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias_c = out_proj_bias.contiguous().to(torch.float32)

        stride_y_b, stride_y_l, stride_y_h = y.stride()
        stride_w_o, stride_w_l, stride_w_i = out_proj_weight_c.stride()
        stride_out_b, stride_out_l, stride_out_h = out.stride()

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y, out_proj_weight_c, out_proj_bias_c, out,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_o, stride_w_l, stride_w_i,
            stride_out_b, stride_out_l, stride_out_h
        )

        return out


def run(*args):
    return ModelNew()(*args)
