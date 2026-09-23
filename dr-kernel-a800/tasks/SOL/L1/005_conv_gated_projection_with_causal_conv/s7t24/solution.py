import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx = F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# BCx: (B, 3H, L), semantics: for each j in [0..3H-1], l: BCx[b,j,l] = sum_i in_proj_weight[j,i,l] * x[b,l,i] + bias[j]
@triton.jit
def in_proj_kernel(
    x_ptr,                   # *float32, input x (B, L, H)
    w_ptr,                   # *float32, weight (3H, H, L)
    b_ptr,                   # *float32, bias (3H)
    out_ptr,                 # *float32, output BCx (B, 3H, L)
    B, L, H,                 # sizes
    stride_x_b, stride_x_l, stride_x_h,        # strides for x (B, L, H)
    stride_w_j, stride_w_i, stride_w_l,        # strides for w (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l   # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, L_tiles, B), where J = 3H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)    # [64], j indices
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128], l indices

    J = 3 * H
    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Accumulator for (j, l) tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over input channels i in [0..H-1]
    # We accumulate x[b, l, i] * w[j, i, l] for all j in tile and l in tile
    for i in range(0, H):
        # Load x[b, l, i]: shape (64,128)
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + i * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load w[j, i, l]: shape (64,128)
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + i * stride_w_i + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += x_vals * w_vals

    # Add bias per j
    bias_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store to out[b, j, l]
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk3: split BCx (B, 3H, L) into three (B, H, L): outB, outC, outX
# We pass pointers for each output and copy the corresponding feature slices directly.
@triton.jit
def chunk3_kernel(
    inp_ptr,     # *float32, input BCx (B, 3H, L)
    outB_ptr,    # *float32, output B_tensor (B, H, L)
    outC_ptr,    # *float32, output C_tensor (B, H, L)
    outX_ptr,    # *float32, output x_proj (B, H, L)
    B, L, H,     # sizes
    stride_inp_b, stride_inp_j, stride_inp_l,    # strides for inp (B, 3H, L)
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


# 3) Element-wise gating: out = a * b, a: (B, L, H), b: (B, L, H)
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
# Causal padding: we pad on the sequence dimension by K-1 (K=4) inside the kernel via masked loads
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, input Bx (B, H, L) contiguous
    w_ptr,                 # *float32, weight (H, H, 4) contiguous
    bias_ptr,              # *float32, bias (H)
    out_ptr,               # *float32, output (B, H, L) contiguous
    B, L, H,               # sizes
    stride_Bx_b, stride_Bx_h, stride_Bx_l,   # strides for Bx (B, H, L)
    stride_w_ho, stride_w_hi, stride_w_k,    # strides for w (H, H, 4)
    stride_out_b, stride_out_h, stride_out_l # strides for out (B, H, L)
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

    # Accumulator for (ho, l) tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel ho, sum over input channel hi and kernel k
    for ho in range(0, H):
        # Initialize bias for ho
        b_val = tl.load(bias_ptr + ho)

        # Loop over input channels hi
        for hi in range(0, H):
            # Loop over k in [0..3] (kernel_size=4)
            for k in range(0, 4):
                # Causal padding: use Bx[b, hi, l+k]
                # If l+k >= L, masked load returns 0.0
                l_padded = l_offsets + k
                mask_l_padded = l_padded < L

                bx_ptrs = Bx_ptr + b * stride_Bx_b + hi * stride_Bx_h + l_padded[None, :] * stride_Bx_l
                bx_vals = tl.load(bx_ptrs, mask=mask_l_padded[None, :], other=0.0)  # (64,128)

                w_ptrs = w_ptr + ho * stride_w_ho + hi * stride_w_hi + k * stride_w_k
                w_val = tl.load(w_ptrs)  # scalar

                acc += bx_vals * w_val

        acc += b_val

    # Store conv_out[b, ho, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y_T (B, L, H) -> out (B, L, H) with weight (H, L, H) and bias (H)
# This implements F.linear(y_T, out_proj_weight, out_proj_bias) with shapes:
# y_T: (B, L, H), out_proj_weight: (H, L, H), bias: (H). Output: (B, L, H).
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y_T (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,      # strides for y_T (B, L, H)
    stride_w_h, stride_w_l, stride_w_h2,     # strides for w (H, L, H)
    stride_out_b, stride_out_l, stride_out_h # strides for out (B, L, H)
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

    # Accumulator for (h, l) tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h, sum over input channels hi (which here is h, given weight shape (H, L, H))
    # This corresponds to: out[b, l, h] = sum over hi of y_T[b, l, hi] * w[hi, l, h] + bias[h]
    # Note: weight shape is (H, L, H). In PyTorch, F.linear with weight (in_features, out_features) here
    # means out_features = H, in_features = L*H, but we must use the provided shape. Given original code uses
    # weight (H, L, H), PyTorch interprets it as (in_features=H, out_features=H), producing output (B, H).
    # To match the provided reference model, we implement the operation as out[b, l, h] = sum_hi y_T[b, l, hi] * w[hi, l, h] + bias[h].
    # We'll iterate hi in [0..H-1].
    for hi in range(0, H):
        # Load y_T[b, l, hi] for tile
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hi * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load w[hi, l, h] for tile
        w_ptrs = w_ptr + hi * stride_w_h + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_h2
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128), broadcasting across h

        acc += y_vals * w_vals

    # Add bias[h]
    bias_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store out[b, l, h]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
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
        # Ensure float32 and contiguous for predictable strides
        B, L, H = x.shape
        device = x.device

        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        # 1) In-projection BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), dtype=torch.float32, device=device)
        J = 3 * H
        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2)
        )

        # 2) Chunk BCx into B_tensor, C_tensor, x_proj
        B_tensor = torch.empty((B, H, L), dtype=torch.float32, device=device)
        C_tensor = torch.empty((B, H, L), dtype=torch.float32, device=device)
        x_proj = torch.empty((B, H, L), dtype=torch.float32, device=device)

        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx, B_tensor, C_tensor, x_proj,
            B, L, H,
            BCx.stride(0), 1, BCx.stride(2),   # stride for feature dim (j) = 1 because BCx is contiguous in feature and last dim
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2)
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, H, L), dtype=torch.float32, device=device)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2)
        )

        # 4) Grouped causal conv: conv_out = conv1d(Bx, conv_weight, conv_bias, groups=H, kernel_size=4)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=device)

        # Prepare Bx_padded by padding on sequence dimension (K-1 = 3) via kernel (no F.pad)
        # The kernel handles masked loads for causal padding.
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)
        )

        # 5) Output gating: y = C_tensor * conv_out
        y = torch.empty((B, H, L), dtype=torch.float32, device=device)
        gate_mul_kernel[grid_gate](
            C_tensor, conv_out, y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2)
        )

        # 6) Final out-projection: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, L, H)
        # We need y_T shape (B, L, H). y is (B, H, L), so transpose last two dims
        y_T = y.transpose(1, 2).contiguous()  # (B, L, H)
        output = torch.empty((B, L, H), dtype=torch.float32, device=device)

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y_T, out_proj_weight, out_proj_bias, output,
            B, L, H,
            y_T.stride(0), y_T.stride(1), y_T.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2)
        )

        return output


def run(*args):
    return ModelNew()(*args)
