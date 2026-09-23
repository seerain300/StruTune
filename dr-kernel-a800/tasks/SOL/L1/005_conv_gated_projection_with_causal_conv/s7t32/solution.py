import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: x -> BCx (B, 3H, L)
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H,)
# BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,                  # *float32, input x (B, L, H)
    w_ptr,                 # *float32, weight (3H, H, L)
    b_ptr,                 # *float32, bias (3H,)
    out_ptr,               # *float32, output BCx (B, 3H, L)
    B, L, H,               # sizes
    stride_x_b, stride_x_l, stride_x_h,      # strides for x (B, L, H)
    stride_w_o, stride_w_i, stride_w_l,      # strides for w (3H, H, L)
    stride_out_b, stride_out_j, stride_out_l  # strides for out (B, 3H, L)
):
    # Grid: (J_tiles, L_tiles, B), where J = 3H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    J = 3 * H
    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # For each output channel j in [0, 3H)
    for j in range(0, J):
        # Compute input channel i = j % H and corresponding L index
        i = j % H
        # Load x[b, l, i]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + i * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load weight w[j, i, l]
        w_ptrs = w_ptr + j * stride_w_o + i * stride_w_i + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        # Load bias for j
        b_val = tl.load(b_ptr + j)  # scalar

        # Accumulate
        out_vals = x_vals * w_vals + b_val

        # Store to out[b, j, l]
        out_ptrs = out_ptr + b * stride_out_b + j * stride_out_j + l_offsets[None, :] * stride_out_l
        tl.store(out_ptrs, out_vals, mask=mask)


# 2) Chunk BCx (B, 3H, L) into B (B, L, H), C (B, L, H), X_proj (B, L, H)
# We split along feature dimension: the first H are B, next H are C, last H are X_proj
@triton.jit
def chunk3_kernel(
    inp_ptr,                # *float32, input BCx (B, 3H, L)
    outB_ptr,               # *float32, output B (B, L, H)
    outC_ptr,               # *float32, output C (B, L, H)
    outX_ptr,               # *float32, output X_proj (B, L, H)
    B, L, H,                # sizes
    stride_inp_b, stride_inp_j, stride_inp_l,     # strides for inp (B, 3H, L)
    stride_out_b, stride_out_l, stride_out_h      # strides for outputs (B, L, H)
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

    # Copy B from inp[b, j=0..H-1, l]
    for ho in range(0, H):
        in_ptrs_B = inp_ptr + b * stride_inp_b + ho * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_B = tl.load(in_ptrs_B, mask=mask, other=0.0)  # (64,128)
        out_ptrs_B = outB_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, vals_B, mask=mask)

    # Copy C from inp[b, j=H..2H-1, l]
    for ho in range(0, H):
        in_ptrs_C = inp_ptr + b * stride_inp_b + (H + ho) * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_C = tl.load(in_ptrs_C, mask=mask, other=0.0)  # (64,128)
        out_ptrs_C = outC_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_C, vals_C, mask=mask)

    # Copy X_proj from inp[b, j=2H..3H-1, l]
    for ho in range(0, H):
        in_ptrs_X = inp_ptr + b * stride_inp_b + (2 * H + ho) * stride_inp_j + l_offsets[None, :] * stride_inp_l
        vals_X = tl.load(in_ptrs_X, mask=mask, other=0.0)  # (64,128)
        out_ptrs_X = outX_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_X, vals_X, mask=mask)


# 3) Element-wise gating: out = a * b
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
# Output conv_out: (B, H, L), stride=1, padding=K-1=3, groups=H (depthwise)
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

    # For each output channel h, sum over its input channel and kernel positions
    for ho in range(0, H):
        # inp[b, ho, l]
        in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_offsets[None, :] * stride_Bx_l
        in_vals = tl.load(in_ptrs, mask=mask, other=0.0)  # (64,128)

        # Accumulate conv along kernel positions k in [0, 4)
        for k in range(0, 4):
            # weight w[ho, ho, k] scalar, groups=H means output channel = input channel
            w_ptrs = w_ptr + ho * stride_w_o + ho * stride_w_i + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            acc += w_val * in_vals  # broadcast w_val to (64,128)

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: y (B, L, H) -> out (B, L, H) with weight (H, L, 3H)
# In our setup, we will pass a weight shaped (H, L, 3H) to represent [B, C, X_proj]
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, 3H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,             # sizes
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_i, stride_w_l, stride_w_j,    # strides for w (H, L, 3H)
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

    # For each output channel h, sum over 3H input channels (B, C, X_proj)
    J = 3 * H
    for j in range(0, J):
        i = j % H  # input channel index
        # y[b, l, i]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + i * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (64,128)

        # weight w[i, l, j]
        w_ptrs = w_ptr + i * stride_w_i + l_offsets[None, :] * stride_w_l + j * stride_w_j
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64,128)

        acc += y_vals * w_vals

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, L, H = x.shape
        # 1) In-projection: BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        in_proj_kernel_B[(triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2)
        )

        # 2) Chunk BCx into B, C, X_proj (B, L, H)
        B_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        C_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        X_proj = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        chunk3_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            BCx, B_tensor, C_tensor, X_proj,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2)
        )

        # 3) Element-wise gating: Bx = B * X_proj (B, L, H)
        Bx = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        gate_mul_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            B_tensor, X_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            X_proj.stride(0), X_proj.stride(1), X_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2)
        )

        # 4) Grouped causal 1D convolution: conv_out (B, H, L), kernel_size=4, groups=H
        conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        conv1d_grouped_causal_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2)
        )

        # 5) Output gating: Y = C * conv_out (B, H, L)
        Y = torch.empty((B, H, L), device=x.device, dtype=torch.float32)
        gate_mul_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            C_tensor, conv_out, Y,
            B, L, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2)
        )

        # Transpose to (B, L, H)
        Y_t = Y.transpose(1, 2).contiguous()  # (B, L, H)

        # 6) Final out-projection: F.linear(Y_t, out_proj_weight, out_proj_bias)
        # out_proj_weight is (H, L, 3H) representing [B, C, X_proj]
        output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        out_proj_kernel[(triton.cdiv(H, 64), triton.cdiv(L, 128), B)](
            Y_t, out_proj_weight, out_proj_bias, output,
            B, L, H,
            Y_t.stride(0), Y_t.stride(1), Y_t.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1), out_proj_weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2)
        )

        return output


def run(*args):
    return ModelNew()(*args)
