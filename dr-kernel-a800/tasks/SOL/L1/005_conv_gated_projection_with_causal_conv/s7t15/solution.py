import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: computes BCx of shape (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr, in_w_ptr, bias_ptr, BCx_ptr,  # pointers
    B, L, H,                            # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x (B, L, H)
    stride_w_o, stride_w_h, stride_w_l,   # strides for in_w (3H, H, L)
    stride_BC_b, stride_BC_j, stride_BC_l  # strides for BCx (B, 3H, L)
):
    # Grid: (J_tiles, B, L_tiles) where J = 3H
    j_block = tl.program_id(0)  # feature tile
    b = tl.program_id(1)        # batch
    l_block = tl.program_id(2)  # seq tile

    J = 3 * H
    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    # Compute output acc (64, 128) for each j in tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each j, compute dot product over L: x[b, l, h] * w[j, h, l]
    for jj in range(0, J):
        # Scalar bias for this output channel j
        bias_val = tl.load(bias_ptr + jj)
        # Load x[b, l, h] for all l in tile
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + jj * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # (1,128), broadcast along jj

        # Load weight w[j, h, l] for all l in tile: h = jj % H
        h_idx = jj % H
        w_ptrs = in_w_ptr + jj * stride_w_o + h_idx * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (1,128)
        acc += x_vals * w_vals  # broadcast multiply

    # Add bias (broadcast across l)
    acc += bias_val

    # Store to BCx[b, j, l]
    out_ptrs = BCx_ptr + b * stride_BC_b + jj * stride_BC_j + l_offsets[None, :] * stride_BC_l
    tl.store(out_ptrs, acc, mask=mask)


# 2) Chunk BCx (B, 3H, L) into three (B, L, H) tensors via Triton (B, C, x_proj)
@triton.jit
def chunk3_kernel(
    inp_ptr, outB_ptr, outC_ptr, outX_ptr,  # pointers
    B, L, H,                               # sizes
    stride_inp_b, stride_inp_j, stride_inp_l,   # strides for inp (B, 3H, L)
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
# Causal padding on sequence: effective L_in = L + 3
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

    H = H  # output channels
    h_offsets = h_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    K = 4
    # For each output channel ho, sum over its input channel ho and kernel positions
    for ho in range(0, H):
        # Effective input length due to causal padding: L_in = L + 3
        L_in = L + 3
        # For each kernel position k in [0, 3], input index idx = l + 3 - k
        for k in range(0, K):
            idx = l_offsets[None, :] + 3 - k  # (1,128)
            in_bounds = (idx >= 0) & (idx < L_in) & mask_l[None, :]
            # Bx[b, ho, idx] with causal padding: idx < L -> normal, else zero
            in_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + idx * stride_Bx_l
            in_vals = tl.load(in_ptrs, mask=in_bounds, other=0.0)  # (1,128)

            # weight w[ho, ho, k] scalar since groups=H => output channel = input channel
            w_ptrs = w_ptr + ho * stride_w_ho + ho * stride_w_hi + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar

            acc += w_val * in_vals  # broadcast w_val to (64,128)

    # Add bias per output channel ho
    bias_val = tl.load(bias_ptr + ho)  # scalar
    acc += bias_val

    # Store to out[b, ho, l]
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
    stride_y_b, stride_y_l, stride_y_h,    # strides for y (B, L, H)
    stride_w_ho, stride_w_li, stride_w_hj, # strides for w (H, L, H)
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

    # For each output channel h_j, compute sum over li and hj
    for hj in range(0, H):
        # y[b, l, hj]
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + hj * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)  # (1,128), broadcast along hj

        # For each li in tile, accumulate over input channels ho
        for li in range(0, H):
            # weight w[ho, li, hj] but since output channel is hj and input channel is li, we can simplify
            w_ptrs = w_ptr + hj * stride_w_ho + li * stride_w_li + hj * stride_w_hj
            w_val = tl.load(w_ptrs)  # scalar
            acc += w_val * y_vals  # broadcast w_val

    # Add bias per output channel hj
    b_val = tl.load(b_ptr + hj)  # scalar
    acc += b_val

    # Store to out[b, l, hj]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets[None, :] * stride_out_l + hj * stride_out_h
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous for Triton
        B, L, H = x.shape
        x = x.contiguous().to(torch.float32)

        # 1) In-projection: BCx of shape (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        in_proj_kernel_B[( (3 * H + 63) // 64, B, (L + 127) // 128 )](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) Transpose BCx to (B, L, 3H) and chunk into B, C, x_proj via Triton
        BCx_T = BCx.transpose(-1, -2)  # (B, L, 3H), contiguous
        B_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        C_tensor = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        x_proj = torch.empty((B, L, H), device=x.device, dtype=torch.float32)

        chunk3_kernel[( (H + 63) // 64, (L + 127) // 128, B )](
            BCx_T, B_tensor, C_tensor, x_proj,
            B, L, H,
            BCx_T.stride(0), BCx_T.stride(2), BCx_T.stride(1),
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        gate_mul_kernel[( (H + 63) // 64, (L + 127) // 128, B )](
            B_tensor, x_proj, Bx,
            B, L, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Grouped causal 1D convolution:
        # Build Bx_padded of shape (B, H, L_in) implicitly via masked loads in Triton
        Bx_contig = Bx.contiguous()  # (B, L, H) -> (B, H, L) by transpose
        conv_out = torch.empty((B, H, L), device=x.device, dtype=torch.float32)

        conv1d_grouped_causal_kernel[( (H + 63) // 64, (L + 127) // 128, B )](
            Bx_contig, conv_weight, conv_bias, conv_out,
            B, L, H,
            Bx_contig.stride(0), Bx_contig.stride(1), Bx_contig.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=4, num_stages=2
        )

        # 5) Output gating: y = C_tensor * conv_out
        y = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        for ho in range(H):  # simple PyTorch for correctness; could be done Triton too, but we keep it light
            y[:, :, ho] = C_tensor[:, :, ho] * conv_out[:, ho, :]

        # 6) Final out-projection: y (B, L, H) -> output (B, L, H)
        output = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        out_proj_kernel[( (H + 63) // 64, (L + 127) // 128, B )](
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
