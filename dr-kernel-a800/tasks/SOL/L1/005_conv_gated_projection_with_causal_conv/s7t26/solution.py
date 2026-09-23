import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: compute BCx of shape (B, 3H, L) using F.linear(x, in_proj_weight, in_proj_bias)
# We implement this as a Triton kernel that, for each (b, j, l), accumulates over H:
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *float32, input x (B, L, H) contiguous
    in_proj_w_ptr,         # *float32, in_proj_weight (3H, H, L) contiguous
    in_proj_b_ptr,         # *float32, in_proj_bias (3H) contiguous
    BCx_ptr,               # *float32, output BCx (B, 3H, L) contiguous
    B, L, H,               # sizes
    stride_x_b, stride_x_l, stride_x_h,     # strides for x (B, L, H)
    stride_w_j, stride_w_h, stride_w_l,     # strides for in_proj_w (3H, H, L)
    stride_BCx_b, stride_BCx_j, stride_BCx_l   # strides for BCx (B, 3H, L)
):
    # Grid: (3H tiles, L tiles, B)
    j = tl.program_id(0)  # output channel in 3H
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128]
    mask_l = l_offsets < L

    # Accumulator for output at (b, j, l)
    acc = tl.zeros((128,), dtype=tl.float32)

    # Accumulate over H
    for hi in range(0, H):
        # x[b, l, hi]
        x_ptrs = x_ptr + b * stride_x_b + l_offsets * stride_x_l + hi * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l, other=0.0)  # (128,)
        # weight[j, hi, l]
        w_ptrs = in_proj_w_ptr + j * stride_w_j + hi * stride_w_h + l_offsets * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask_l, other=0.0)  # (128,)
        acc += x_vals * w_vals

    # add bias[j]
    bias_val = tl.load(in_proj_b_ptr + j)
    acc += bias_val

    # store to BCx[b, j, l]
    out_ptrs = BCx_ptr + b * stride_BCx_b + j * stride_BCx_j + l_offsets * stride_BCx_l
    tl.store(out_ptrs, acc, mask=mask_l)


# 2) Grouped causal 1D convolution with gating and output gating (fused with out-projection)
# Input: BCx (B, 3H, L)
# Output: final_out (B, L, H)
# We:
# - Transpose BCx to (B, L, 3H) inside kernel via pointer math
# - Split into B_tensor, C_tensor, x_proj (by copying slices)
# - Compute Bx = B_tensor * x_proj
# - Pad Bx left by 3 for causal (using pad_left3_kernel to produce Bx_padded)
# - Conv1d groups=H, kernel_size=4, stride=1, with conv_bias
# - y = C_tensor * conv_out (elementwise)
# - out = F.linear(y, out_proj_weight, out_proj_bias)
@triton.jit
def conv_grouped_with_gating_and_out_kernel(
    BCx_ptr,            # *float32, input BCx (B, 3H, L) contiguous
    w_ptr,              # *float32, conv weight (H, H, 4) contiguous
    conv_bias_ptr,      # *float32, conv bias (H) contiguous
    out_proj_w_ptr,     # *float32, out-proj weight (H, L, H) contiguous
    out_bias_ptr,       # *float32, out-proj bias (H) contiguous
    final_out_ptr,      # *float32, output (B, L, H) contiguous
    B, L, H,            # sizes
    stride_BCx_b, stride_BCx_j, stride_BCx_l,   # strides for BCx (B, 3H, L)
    stride_w_o, stride_w_i, stride_w_k,         # strides for weight (H, H, 4)
    stride_out_b, stride_out_l, stride_out_h   # strides for final_out (B, L, H)
):
    # Grid: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)  # for output channels
    l_block = tl.program_id(1)  # for L tiles
    b = tl.program_id(2)

    h_offsets = h_block * 64 + tl.arange(0, 64)      # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)    # [128]

    mask_h = h_offsets < H
    mask_l = l_offsets < L
    mask = mask_h[:, None] & mask_l[None, :]

    # Compute B_tensor, C_tensor, x_proj by copying slices from BCx
    # Transpose BCx to (B, L, 3H) via pointer arithmetic, then chunk into three parts.
    # Note: Since we need B_tensor = BCx[b, :, H:H*2], C_tensor = BCx[b, :, 2H:3H], x_proj = BCx[b, :, 0:H],
    # we directly copy them into output tensors (B, L, H) with appropriate strides.

    # Allocate outputs for B_tensor, C_tensor, x_proj (host code will pass these pointers)
    # We'll compute B_tensor, C_tensor, x_proj here, then use them to form Bx and y.
    # However, to avoid extra allocations, we will compute B_tensor, C_tensor, x_proj inside this kernel
    # by reading from BCx and writing to temporary outputs. For simplicity, we assume host allocates
    # B_tensor, C_tensor, x_proj tensors of shape (B, L, H). We implement chunking by writing them.
    # We'll implement chunking via copying from BCx using pointer math.

    # Copy B_tensor = BCx[b, :, H:H*2] -> (B, L, H)
    for ho in range(0, H):
        # For each output channel j in [H, 2H)
        j = H + ho
        in_ptrs = BCx_ptr + b * stride_BCx_b + j * stride_BCx_j + l_offsets * stride_BCx_l
        out_ptrs_B = B_tensor_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_B, tl.load(in_ptrs, mask=mask_l, other=0.0), mask=mask_l)

    # Copy C_tensor = BCx[b, :, 2H:3H] -> (B, L, H)
    for ho in range(0, H):
        j = 2 * H + ho
        in_ptrs = BCx_ptr + b * stride_BCx_b + j * stride_BCx_j + l_offsets * stride_BCx_l
        out_ptrs_C = C_tensor_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_C, tl.load(in_ptrs, mask=mask_l, other=0.0), mask=mask_l)

    # Copy x_proj = BCx[b, :, 0:H] -> (B, L, H)
    for ho in range(0, H):
        in_ptrs = BCx_ptr + b * stride_BCx_b + ho * stride_BCx_j + l_offsets * stride_BCx_l
        out_ptrs_X = x_proj_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs_X, tl.load(in_ptrs, mask=mask_l, other=0.0), mask=mask_l)

    # Element-wise gating: Bx = B_tensor * x_proj
    for ho in range(0, H):
        a_ptrs = B_tensor_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        b_ptrs = x_proj_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        out_ptrs = Bx_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        a_vals = tl.load(a_ptrs, mask=mask_l, other=0.0)
        b_vals = tl.load(b_ptrs, mask=mask_l, other=0.0)
        tl.store(out_ptrs, a_vals * b_vals, mask=mask_l)

    # Causal padding for Bx: pad left by 3
    # Create Bx_padded (B, L+3, H) with zeros. We'll implement pad via a Triton kernel that copies.
    # Allocate Bx_padded
    Bx_padded = tl.zeros((B, L + 3, H), dtype=tl.float32)  # symbolic, but in Triton we launch a kernel
    pad_left3_kernel(Bx_ptr, Bx_padded, B, L, H)

    # Grouped conv1d on Bx_padded (B, L+3, H) with groups=H, kernel_size=4
    conv_out = tl.zeros((B, H, L), dtype=tl.float32)  # symbolic, implement in kernel

    # Now compute conv_out in Triton:
    # We'll implement the conv as a separate kernel conv1d_grouped_kernel that takes Bx_padded and weight.
    # Here we avoid using conv1d and implement explicit conv loops.

    # Implement grouped conv: for each h in [0..H), compute conv_out[b, h, l] = sum over k=0..3 of
    # Bx_padded[b, l+1+k, h] * w[h, h, k] + conv_bias[h]
    for ho in range(0, H):
        h_out = ho
        # Initialize acc for this (b, h_out)
        acc = tl.zeros((L,), dtype=tl.float32)
        for k in range(0, 4):
            # For grouped conv, each output channel uses its own input channel
            in_ptrs_k = Bx_padded + b * (L + 3) * H + (l_offsets + 1 + k) * H + h_out
            # For vectorized loads, we need to build per-l offsets for in_ptrs_k. Triton does element-wise indexing.
            # We can load a vector by iterating l_offsets:
            for l_idx in range(0, L):
                in_val = tl.load(in_ptrs_k + l_idx, mask=False, other=0.0)  # scalar load
                w_ptrs = w_ptr + h_out * stride_w_o + h_out * stride_w_i + k * stride_w_k
                w_val = tl.load(w_ptrs)  # scalar
                acc[l_idx] += in_val * w_val
        # Add bias
        bias_val = tl.load(conv_bias_ptr + h_out)
        acc += bias_val

        # Store conv_out[b, h_out, l]
        out_ptrs = conv_out_ptr + b * (H * L) + h_out * L + l_offsets
        tl.store(out_ptrs, acc, mask=mask_l)

    # Output gating: y = C_tensor * conv_out
    for ho in range(0, H):
        C_vals = tl.load(C_tensor_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h, mask=mask_l, other=0.0)  # (L,)
        conv_vals = tl.load(conv_out_ptr + b * (H * L) + ho * L + l_offsets, mask=mask_l, other=0.0)  # (L,)
        y_vals = C_vals * conv_vals
        out_ptrs = y_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs, y_vals, mask=mask_l)

    # Final out-projection: out = F.linear(y, out_proj_weight, out_proj_bias)
    # We need to compute out_proj for each ho in [0..H): out[b, l, ho] = sum_l' y[b, l', ho] * out_proj_w[ho, l', ho] + out_bias[ho]
    # Implement this per ho in Triton.
    for ho in range(0, H):
        # Initialize final_out[b, l, ho] = 0
        final_vals = tl.zeros((L,), dtype=tl.float32)
        # For each l' in [0..L-1]
        for l_prime in range(0, L):
            # y[b, l', ho]
            y_ptr_cur = y_ptr + b * stride_out_b + l_prime * stride_out_l + ho * stride_out_h
            y_val = tl.load(y_ptr_cur)
            # out_proj_w[ho, l', ho]
            w_ptr_cur = out_proj_w_ptr + ho * stride_w_o + l_prime * stride_w_l + ho * stride_w_i
            w_val = tl.load(w_ptr_cur)
            final_vals[l_prime] += y_val * w_val
        # Add bias
        bias_val = tl.load(out_bias_ptr + ho)
        final_vals += bias_val
        # Store final_out[b, l, ho]
        out_ptrs = final_out_ptr + b * stride_out_b + l_offsets * stride_out_l + ho * stride_out_h
        tl.store(out_ptrs, final_vals, mask=mask_l)


# Helper Triton kernel: pad left by 3 elements of Bx (B, L, H) to produce Bx_padded (B, L+3, H) with zeros in the first 3 positions.
@triton.jit
def pad_left3_kernel(
    Bx_ptr,              # *float32, input Bx (B, L, H) contiguous
    Bx_padded_ptr,       # *float32, output Bx_padded (B, L+3, H) contiguous
    B, L, H              # sizes
):
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    h_block = tl.program_id(2)

    l_offsets = l_block * 128 + tl.arange(0, 128)    # [128]
    h_offsets = h_block * 64 + tl.arange(0, 64)      # [64]

    mask_l = l_offsets < L
    mask_h = h_offsets < H
    mask = mask_l[:, None] & mask_h[None, :]

    # For positions 3..L+2, copy Bx[b, l-3, h] into Bx_padded[b, l, h]
    for l_idx in range(0, 128):
        if l_offsets[l_idx] + 3 < L + 3:
            in_ptrs = Bx_ptr + b * (L * H) + (l_offsets[l_idx] + 3) * H + h_offsets[None, :]
            out_ptrs = Bx_padded_ptr + b * ((L + 3) * H) + l_offsets[l_idx] * H + h_offsets[None, :]
            tl.store(out_ptrs, tl.load(in_ptrs, mask=mask, other=0.0), mask=mask)

    # For positions 0..2, set zeros
    for l_idx in range(0, 3):
        out_ptrs = Bx_padded_ptr + b * ((L + 3) * H) + l_idx * H + h_offsets[None, :]
        zeros = tl.zeros((64,), dtype=tl.float32)
        tl.store(out_ptrs, zeros, mask=mask_h[None, :])


# Launch ModelNew.forward uses these kernels
class ModelNew(nn.Module):
    def forward(
        self,
        x: torch.Tensor,                 # (B, L, H)
        in_proj_weight: torch.Tensor,    # (3H, H, L)
        in_proj_bias: torch.Tensor,      # (3H)
        conv_weight: torch.Tensor,       # (H, H, 4)
        conv_bias: torch.Tensor,         # (H)
        out_proj_weight: torch.Tensor,   # (H, L, H)
        out_proj_bias: torch.Tensor,     # (H)
    ):
        B, L, H = x.shape
        # Ensure contiguous tensors and float32 for predictable strides
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        # 1) In-projection: compute BCx (B, 3H, L)
        BCx = torch.empty((B, 3 * H, L), dtype=torch.float32, device=x.device)
        # Launch grid over (3H, L tiles, B)
        grid_in = (3 * H, triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2)
        )

        # 2) Fused grouped conv with gating and final out-projection
        # Prepare outputs: B_tensor, C_tensor, x_proj, Bx, conv_out, y, final_out
        B_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        C_tensor = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        x_proj = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        Bx = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)
        y = torch.empty((B, L, H), dtype=torch.float32, device=x.device)
        final_out = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        grid = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv_grouped_with_gating_and_out_kernel[grid](
            BCx, conv_weight, conv_bias, out_proj_weight, out_proj_bias, final_out,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            final_out.stride(0), final_out.stride(1), final_out.stride(2)
        )

        # Return final_out (B, L, H)
        return final_out


# Optional: simple test (not used by evaluator, but useful for local validation)
if __name__ == "__main__":
    # Example with small sizes; actual evaluator will use provided axes.
    B, L, H = 2, 1024, 128
    x = torch.randn(B, L, H, device="cuda", dtype=torch.float32)
    in_proj_weight = torch.randn(3 * H, H, L, device="cuda", dtype=torch.float32)
    in_proj_bias = torch.randn(3 * H, device="cuda", dtype=torch.float32)
    conv_weight = torch.randn(H, H, 4, device="cuda", dtype=torch.float32)
    conv_bias = torch.randn(H, device="cuda", dtype=torch.float32)
    out_proj_weight = torch.randn(H, L, H, device="cuda", dtype=torch.float32)
    out_proj_bias = torch.randn(H, device="cuda", dtype=torch.float32)

    model = ModelNew()
    out = model(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)
    print(out.shape)  # should be (B, L, H)


def run(*args):
    return ModelNew()(*args)
