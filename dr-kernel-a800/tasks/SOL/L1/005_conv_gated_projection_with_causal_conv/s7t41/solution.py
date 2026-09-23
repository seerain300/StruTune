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
    o_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    O = 3 * H  # out_features

    o_offsets = o_block * 64 + tl.arange(0, 64)  # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]

    mask_o = o_offsets < O
    mask_l = l_offsets < L
    mask = mask_o[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output o in [0, 3H)
    for o in range(0, O):
        # Compute dot product over H (in_features)
        dot = tl.zeros((1,), dtype=tl.float32)
        for i in range(0, H):
            x_val = tl.load(x_ptr + b * stride_x_b + i * stride_x_h + l_offsets * stride_x_l)  # (128,)
            w_val = tl.load(w_ptr + o * stride_w_o + i * stride_w_i + l_offsets * stride_w_l)  # (128,)
            dot += x_val * w_val
        # Add bias for this o
        b_val = tl.load(bias_ptr + o)  # scalar
        dot += b_val

        acc += dot * (o_offsets == o)  # broadcast scalar into 64th position where o==o; correct approach: build per o
        # The above line is illustrative; correct approach is to directly assign acc[o] = dot. Triton doesn't support direct indexing by tensor, so we build acc via multiply with mask(o==o), which is always true. However, simpler approach: compute per o directly with vectorized broadcast:
        # We'll do the computation directly into acc per o by loading w and x for that o. Replace previous loop with:
        # For each i in [0,H), load x and w for this o and l_offsets, accumulate. Then store.
        # Implementing the correct accumulation:
        dot_acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, H):
            x_val = tl.load(x_ptr + b * stride_x_b + i * stride_x_h + l_offsets * stride_x_l)  # (128,)
            w_val = tl.load(w_ptr + o * stride_w_o + i * stride_w_i + l_offsets * stride_w_l)  # (128,)
            dot_acc += tl.sum(x_val * w_val, axis=0)  # reduce over 128 lanes to scalar
        # Now dot_acc is a scalar; broadcast add to acc
        acc += dot_acc  # broadcasting scalar to (64,128), but only one o lane should get it. Better: construct a 2D vector with o==o mask and set acc[:, None] to dot_acc; since we have only one o per loop, we can just add dot_acc to all. To set only o lane, we need a mask vector. Triton doesn't support dynamic indexing for assignment; instead, we compute dot_acc once and add to all. This is acceptable because acc will hold the same dot value for all o in this loop. To fix, we compute per o and store directly.

        # Fix: compute per o and store directly into BCx without relying on acc's o lane.
        # We'll implement the correct approach below.

    # The above loop had a logical issue: we were adding the same dot_acc to all o lanes.
    # Correct approach: compute per o and store directly. Since Triton doesn't support direct indexing, we re-implement per o computation and store using a mask for o_offsets == o.
    for o in range(0, O):
        dot_acc = tl.zeros((), dtype=tl.float32)
        for i in range(0, H):
            x_val = tl.load(x_ptr + b * stride_x_b + i * stride_x_h + l_offsets * stride_x_l)  # (128,)
            w_val = tl.load(w_ptr + o * stride_w_o + i * stride_w_i + l_offsets * stride_w_l)  # (128,)
            dot_acc += tl.sum(x_val * w_val, axis=0)  # scalar
        # Add bias
        b_val = tl.load(bias_ptr + o)
        dot_acc += b_val

        # Store BCx[b, o, l] for all l_offsets
        out_ptrs = BCx_ptr + b * stride_BC_b + o * stride_BC_o + l_offsets * stride_BC_l
        tl.store(out_ptrs, dot_acc, mask=mask_o[:, None] & mask_l[None, :])


# 2) Chunk: split BCx (B, L, 3H) -> B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
# Transpose BCx to (B, L, 3H) and copy slices using Triton.
@triton.jit
def chunk3_kernel(
    BCx_ptr,        # *float32, input BCx (B, L, 3H) contiguous
    outB_ptr,       # *float32, output B (B, L, H) contiguous
    outC_ptr,       # *float32, output C (B, L, H) contiguous
    outX_ptr,       # *float32, output x_proj (B, L, H) contiguous
    B, L, H,        # sizes
    stride_BC_b, stride_BC_l, stride_BC_j,   # strides for BCx (B, L, 3H)
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

    # B: slice j = 0..H-1
    inB_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + h_offsets[:, None] * stride_BC_j
    outB_ptrs = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outB_ptrs, tl.load(inB_ptrs, mask=mask, other=0.0), mask=mask)

    # C: slice j = H..2H-1
    inC_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + (H + h_offsets[:, None]) * stride_BC_j
    outC_ptrs = outC_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outC_ptrs, tl.load(inC_ptrs, mask=mask, other=0.0), mask=mask)

    # x_proj: slice j = 2H..3H-1
    inX_ptrs = BCx_ptr + b * stride_BC_b + l_offsets[None, :] * stride_BC_l + (2 * H + h_offsets[:, None]) * stride_BC_j
    outX_ptrs = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(outX_ptrs, tl.load(inX_ptrs, mask=mask, other=0.0), mask=mask)


# 3) Element-wise gating: out = a * b (vectorized), here a=B, b=x_proj
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
# Note: We pass a padded input along the sequence length: Bx_padded (B, H, L + 3) with 3 zeros on the left.
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L + Kpad) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes (note: output L is original L)
    Kpad,                  # int, padding (K-1) = 3
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L + Kpad)
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

    # For each output channel h
    for ho in range(0, H):
        # For each position in L
        for l in range(0, L):
            # Sum over kernel positions k in [0, 4)
            for k in range(0, 4):
                l_src = l + Kpad - k  # causal: pad zeros on left
                # Load input value Bx[b, ho, l_src]; if l_src < 0 or >= L+Kpad, value is 0
                in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_src * stride_Bx_l
                in_val = tl.load(in_ptrs, mask=(l_src >= 0) & (l_src < (L + Kpad)), other=0.0)  # scalar
                # Load weight w[ho, ho, k] scalar (groups=H means output channel = input channel)
                w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
                w_val = tl.load(w_ptrs)
                acc += in_val * w_val  # broadcast to (64,128) by adding scalar to all lanes

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Final out-projection: F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, L, H), out_proj_weight: (H, L, H), out_proj_bias: (H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H) contiguous
    w_ptr,               # *float32, weight (H, L, H) contiguous
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H) contiguous
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,     # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,     # strides for weight (H, L, H)
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

    # For each output channel ho
    for ho in range(0, H):
        acc = tl.zeros((64, 128), dtype=tl.float32)
        # For each l
        for li in range(0, L):
            # For each input channel hi
            for hi in range(0, H):
                y_val = tl.load(y_ptr + b * stride_y_b + li * stride_y_l + hi * stride_y_h)  # scalar
                w_val = tl.load(w_ptr + ho * stride_w_o + li * stride_w_l + hi * stride_w_i)  # scalar
                acc += y_val * w_val
        # Add bias
        b_val = tl.load(b_ptr + ho)
        acc += b_val
        # Store to out[b, ho, li]
        out_ptrs = out_ptr + b * stride_out_b + ho * stride_out_h + l_offsets[None, :] * stride_out_l
        tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure device and dtype
        device = x.device
        B, L, H = x.shape
        x_transposed = x.transpose(1, 2).contiguous()  # (B, H, L), float32
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        # 1) In-projection: BCx of shape (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=device, dtype=torch.float32)
        # Launch Triton kernel
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x_transposed, in_proj_weight, in_proj_bias, BCx,
            B, H, L,
            x_transposed.stride(0), x_transposed.stride(1), x_transposed.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Chunk: split BCx (B, L, 3H) -> B: (B, L, H), C: (B, L, H), x_proj: (B, L, H)
        BCx_t = BCx.transpose(-1, -2).contiguous()  # (B, L, 3H)
        outB = torch.empty((B, L, H), device=device, dtype=torch.float32)
        outC = torch.empty((B, L, H), device=device, dtype=torch.float32)
        outX = torch.empty((B, L, H), device=device, dtype=torch.float32)

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_t, outB, outC, outX,
            B, L, H,
            BCx_t.stride(0), BCx_t.stride(1), BCx_t.stride(2),
            outB.stride(0), outB.stride(1), outB.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, L, H), device=device, dtype=torch.float32)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            outB, outX, Bx,
            B, L, H,
            outB.stride(0), outB.stride(1), outB.stride(2),
            outX.stride(0), outX.stride(1), outX.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: Bx_padded (B, H, L + 3), conv_weight (H, H, 4), conv_bias (H)
        Kpad = 3
        Bx_padded = torch.nn.functional.pad(Bx, (Kpad, 0))  # pad left by 3
        conv_out = torch.empty((B, H, L), device=device, dtype=torch.float32)

        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx_padded, conv_weight, conv_bias, conv_out,
            B, L, H,
            Kpad,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C * conv_out
        y = torch.empty((B, L, H), device=device, dtype=torch.float32)
        grid_gate2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate2](
            outC, conv_out, y,
            B, L, H,
            outC.stride(0), outC.stride(1), outC.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4, num_stages=2
        )

        # 6) Final out-projection: F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, L, H), device=device, dtype=torch.float32)
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
