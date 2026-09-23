import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: x_transposed (B, H, L) and weight (3H, H, L) -> BCx (B, 3H, L)
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
    # Grid: (O_tiles, L_tiles, B) where O = 3H
    O = 3 * H
    o_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    o_offsets = o_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_o = o_offsets < O
    mask_l = l_offsets < L
    mask = mask_o[:, None] & mask_l[None, :]

    # For each output feature o in [0, 3H)
    for o in range(0, O):
        # Compute output channel index i = o % H
        i = o % H

        # Accumulator
        acc = tl.zeros((64, 128), dtype=tl.float32)

        # Reduction over input features i (actually single i, but kept generic)
        # Here we implement the linear operation for each o: sum over input channels i and L
        # For simplicity, we compute x[b, i, l] and w[o, i, l] and accumulate.
        # Note: Weight shape is (3H, H, L), input x shape is (B, H, L).
        # We need to compute dot over i: sum_i x[b, i, l] * w[o, i, l] + bias[o]
        for ii in range(0, H):
            # x[b, ii, l]
            x_vals = tl.load(
                x_ptr + b * stride_x_b + ii * stride_x_h + l_offsets[None, :] * stride_x_l,
                mask=mask_l[None, :],
                other=0.0,
            )  # (1,128), broadcast later
            # w[o, ii, l]
            w_vals = tl.load(
                w_ptr + o * stride_w_o + ii * stride_w_i + l_offsets[None, :] * stride_w_l,
                mask=mask_l[None, :],
                other=0.0,
            )  # (1,128)
            # Accumulate
            acc += x_vals * w_vals

        # Add bias
        bias_val = tl.load(bias_ptr + o, mask=mask_o[o], other=0.0)
        acc += bias_val  # broadcast along (128)

        # Store to BCx[b, o, l]
        out_ptrs = BCx_ptr + b * stride_BC_b + o * stride_BC_o + l_offsets[None, :] * stride_BC_l
        tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk: split BCx (B, 3H, L) into B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
@triton.jit
def chunk3_kernel(
    BCx_ptr,        # *float32, input BCx (B, 3H, L) contiguous
    outB_ptr,       # *float32, output B (B, L, H) contiguous
    outC_ptr,       # *float32, output C (B, L, H) contiguous
    outX_ptr,       # *float32, output x_proj (B, L, H) contiguous
    B, L, H,        # sizes
    stride_BC_b, stride_BC_o, stride_BC_l,   # strides for BCx (B, 3H, L)
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
# Input Bx: (B, H, L) (from x_transposed), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L), causal padding on the fly for each k
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_o, stride_w_i, stride_w_k,      # strides for weight (H, H, 4)
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

    # Initialize accumulator
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over kernel positions k in [0, 4)
    K = 4
    for k in range(0, K):
        # causal index: l_in = l + 3 - k
        l_in = l_offsets + (K - 1 - k)  # pad_left = K - 1 for causal
        # mask for valid l_in
        mask_l_in = l_in < L

        # For each input channel h, accumulate: sum_i weight[h, i, k] * Bx[b, i, l_in]
        for i in range(0, H):
            # Load input Bx[b, i, l_in]
            in_ptrs = Bx_ptr + b * stride_Bx_b + i * stride_Bx_h + l_in[None, :] * stride_Bx_l
            in_vals = tl.load(in_ptrs, mask=mask_l_in[None, :], other=0.0)  # (1,128)

            # Load weight[h, i, k]
            w_ptrs = w_ptr + h_offsets[:, None] * stride_w_o + i * stride_w_i + k * stride_w_k
            w_vals = tl.load(w_ptrs, mask=mask_h[:, None], other=0.0)  # (64,1)
            # Broadcast multiply
            acc += w_vals * in_vals  # (64,128)

    # Add bias
    bias_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Final out-projection: y (B, L, H) and weight (H, L, H) -> output (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H) contiguous
    w_ptr,               # *float32, weight (H, L, H) contiguous
    b_ptr,               # *float32, bias (H) contiguous
    out_ptr,             # *float32, output (B, L, H) contiguous
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,   # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,   # strides for weight (H, L, H)
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

    # Accumulator for output over H (each output channel)
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h
    for ho in range(0, H):
        # Load y[b, l, ho]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + ho * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load weight[ho, l, i] and accumulate: sum_i w[ho, l, i] * y[l, i]
        # weight layout is (H, L, H): we index by (ho, l, i)
        w_ptrs = w_ptr + ho * stride_w_o + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_i
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += w_vals * y_vals  # (64,128)

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, l, h]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask)


# Entry point ModelNew
class ModelNew(nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # x: (B, L, H); ensure float32 and contiguous
        B, L, H = x.shape
        device = x.device

        # 1) In-projection: transpose x for F.linear signature
        x_t = x.transpose(1, 2).contiguous()  # (B, H, L)
        x_t = x_t.to(torch.float32)

        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()

        BCx = torch.empty((B, 3 * H, L), dtype=torch.float32, device=device)

        # Launch in-projection Triton kernel
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x_t, in_proj_weight, in_proj_bias, BCx,
            B, H, L,
            x_t.stride(0), x_t.stride(1), x_t.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Chunk BCx (B, 3H, L) into B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
        B_t = torch.empty((B, L, H), dtype=torch.float32, device=device)
        C_t = torch.empty((B, L, H), dtype=torch.float32, device=device)
        Xp = torch.empty((B, L, H), dtype=torch.float32, device=device)

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx, B_t, C_t, Xp,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            B_t.stride(0), B_t.stride(1), B_t.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_t * Xp
        Bx = torch.empty((B, H, L), dtype=torch.float32, device=device)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_t, Xp, Bx,
            B, L, H,
            B_t.stride(0), B_t.stride(1), B_t.stride(2),
            Xp.stride(0), Xp.stride(1), Xp.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: conv_weight (H, H, 4), conv_bias (H)
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()

        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=device)
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C_t * conv_out (C_t shape is (B, L, H), conv_out shape (B, H, L))
        # We need to align shapes: conv_out transpose to (B, L, H) for elementwise mul
        # conv_out currently (B, H, L), transpose along last two dims to get (B, L, H)
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, L, H)

        y = torch.empty((B, L, H), dtype=torch.float32, device=device)
        grid_gate2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate2](
            C_t, conv_out_T, y,
            B, L, H,
            C_t.stride(0), C_t.stride(1), C_t.stride(2),
            conv_out_T.stride(0), conv_out_T.stride(1), conv_out_T.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4, num_stages=2
        )

        # 6) Final out-projection: y (B, L, H) and out_proj_weight (H, L, H) -> output (B, L, H)
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        output = torch.empty((B, L, H), dtype=torch.float32, device=device)
        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B, L, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
