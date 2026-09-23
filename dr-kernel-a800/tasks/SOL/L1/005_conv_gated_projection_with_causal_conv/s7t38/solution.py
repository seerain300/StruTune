import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx of shape (B, 3H, L)
# x: (B, L, H), w: (3H, H, L) (note: L dimension is last in PyTorch's linear), bias: (3H)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                  # *float32, input x (B, L, H) contiguous
    w_ptr,                  # *float32, in_proj_weight (3H, H, L) contiguous
    b_ptr,                  # *float32, in_proj_bias (3H)
    BCx_ptr,                # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,                # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x (B, L, H)
    stride_w_o, stride_w_i, stride_w_l,   # strides for w (3H, H, L) -> (o, i, l)
    stride_BC_b, stride_BC_o, stride_BC_l # strides for BCx (B, 3H, L)
):
    # Grid: (O_tiles, L_tiles, B) where O=3H
    o_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    O = 3 * H
    o_offsets = o_block * 64 + tl.arange(0, 64)           # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)         # [128]

    mask_o = o_offsets < O
    mask_l = l_offsets < L
    mask = mask_o[:, None] & mask_l[None, :]

    # Accumulator for BCx[b, o, l]
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel o, compute dot over H and L
    for ho in range(0, H):  # iterate over input channels i
        # For o in [0, 3H), the corresponding input channel i = o % H
        i = o_offsets % H  # (64,)
        # Load x[b, l, i] for each l
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + i[:, None] * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load w[o, i, l]
        w_ptrs = w_ptr + o_offsets[:, None] * stride_w_o + i[None, :] * stride_w_i + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += x_vals * w_vals

    # Add bias for each o
    b_vals = tl.load(b_ptr + o_offsets, mask=mask_o, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to BCx[b, o, l]
    out_ptrs = BCx_ptr + b * stride_BC_b + o_offsets[:, None] * stride_BC_o + l_offsets[None, :] * stride_BC_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Element-wise gating: out = a * b (vectorized), where a is B_tensor and b is x_proj
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


# 3) Left-pad along last dimension by pad_len (for causal conv with kernel_size=4, pad=3)
@triton.jit
def pad_left_kernel(
    inp_ptr,         # *float32, input tensor (B, D, T)
    out_ptr,         # *float32, output tensor (B, D, T + pad_len), contiguous
    B, D, T, pad_len,
    stride_inp_b, stride_inp_d, stride_inp_t,
    stride_out_b, stride_out_d, stride_out_t
):
    # Grid: (D_tiles, T_tiles, B)
    d_block = tl.program_id(0)
    t_block = tl.program_id(1)
    b = tl.program_id(2)

    d_offsets = d_block * 64 + tl.arange(0, 64)   # [64]
    t_offsets = t_block * 128 + tl.arange(0, 128) # [128]

    mask_d = d_offsets < D
    mask_t = t_offsets < T
    mask = mask_d[:, None] & mask_t[None, :]

    # For each original time index t, store to out at t + pad_len
    inp_ptrs = inp_ptr + b * stride_inp_b + d_offsets[:, None] * stride_inp_d + t_offsets[None, :] * stride_inp_t
    vals = tl.load(inp_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + b * stride_out_b + d_offsets[:, None] * stride_out_d + (t_offsets[None, :] + pad_len) * stride_out_t
    tl.store(out_ptrs, vals, mask=mask)


# 4) Grouped causal 1D convolution:
# Input Bx_pad: (B, H, Tpad), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, T) where T = Tpad - 3 (for K=4)
@triton.jit
def conv_group_causal_kernel(
    Bx_pad_ptr,              # *float32, input padded tensor (B, H, Tpad) contiguous
    w_ptr,                   # *float32, weight (H, H, 4) contiguous
    b_ptr,                   # *float32, bias (H)
    out_ptr,                 # *float32, output (B, H, T) contiguous
    B, H, Tpad, T,           # sizes: T = Tpad - 3
    stride_Bx_b, stride_Bx_h, stride_Bx_t,   # strides for Bx_pad (B, H, Tpad)
    stride_w_o, stride_w_i, stride_w_k,      # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_t # strides for out (B, H, T)
):
    # Grid: (H_tiles, T_tiles, B)
    h_block = tl.program_id(0)
    t_block = tl.program_id(1)
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    t_offsets = t_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_t = t_offsets < T
    mask = mask_h[:, None] & mask_t[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    K = 4
    for k in range(0, K):
        t_in = t_offsets + k  # (128,)
        valid = (t_in >= 0) & (t_in < Tpad) & mask_t[None, :]  # broadcast over h

        # Load Bx_pad[b, h, t_in]
        in_ptrs = Bx_pad_ptr + b * stride_Bx_b + h_offsets[:, None] * stride_Bx_h + t_in[None, :] * stride_Bx_t
        x_vals = tl.load(in_ptrs, mask=valid, other=0.0)  # (64, 128)

        # Load w[h, h, k]
        w_ptrs = w_ptr + h_offsets[:, None] * stride_w_o + h_offsets[None, :] * stride_w_i + k * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,)

        # Accumulate (grouped: input channel = output channel)
        acc += w_vals[:, None] * x_vals

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store conv_out[b, h, t]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + t_offsets[None, :] * stride_out_t
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H), out_proj_weight (H, L, H), bias (H) -> output (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H) contiguous
    w_ptr,               # *float32, weight (H, L, H) contiguous
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H) contiguous
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_o, stride_w_l, stride_w_i,    # strides for w (H, L, H)
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

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Compute out[b, l, h] = sum_i w[h, l, i] * y[b, l, i] + bias[h]
    for hi in range(0, H):
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hi * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64, 128)

        w_ptrs = w_ptr + hi * stride_w_o + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_i
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        acc += w_vals * y_vals

    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
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
        """
        Triton-only fused implementation of:
          1) in-projection: x -> BCx (B, 3H, L)
          2) chunk: B_tensor, C_tensor, x_proj (each B, L, H) via slicing (no torch.chunk)
          3) gating: Bx = B_tensor * x_proj
          4) causal grouped conv with kernel_size=4 on Bx_pad -> conv_out (B, H, L)
          5) gating: y = C_tensor * conv_out
          6) out-projection: y -> output (B, L, H)
        """
        assert x.dim() == 3, "x must be (B, L, H)"
        B, L, H = x.shape
        # Ensure contiguity and float32
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Slice to get B_tensor, C_tensor, x_proj directly from BCx without torch.chunk
        # BCx has layout (B, 3H, L); we access slices by known indices.
        B_tensor = BCx[:, :H, :].contiguous()         # (B, L, H)
        C_tensor = BCx[:, H:2*H, :].contiguous()      # (B, L, H)
        x_proj = BCx[:, 2*H:, :].contiguous()         # (B, L, H)

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_gm = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gm](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Pad left for causal conv (kernel_size=4 -> pad=3)
        Bx_pad = torch.empty((B, H, L + 3), device=x.device, dtype=torch.float32)
        grid_pad = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        pad_len = 3
        pad_left_kernel[grid_pad](
            Bx, Bx_pad,
            B, H, L, pad_len,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution: conv_out (B, H, L)
        conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv_group_causal_kernel[grid_conv](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, L + 3, L,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C_tensor * conv_out
        y = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid_gm2 = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gm2](
            C_tensor, conv_out, y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=4, num_stages=2
        )

        # 6) Final out-projection: output (B, L, H)
        output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
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


# The following functions are used by the evaluation harness (not part of the Triton-only computation)
@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    model = ModelNew()
    return model(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
