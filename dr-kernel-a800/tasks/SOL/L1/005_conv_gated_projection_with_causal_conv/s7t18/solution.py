import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1) In-projection: BCx = x @ in_proj_weight^T + in_proj_bias
# x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H), BCx: (B, 3H, L)
@triton.jit
def in_proj_kernel_B(
    x_ptr,              # *float32, x (B, L, H) contiguous
    w_ptr,              # *float32, in_proj_weight (3H, H, L) contiguous
    b_ptr,              # *float32, in_proj_bias (3H)
    BCx_ptr,            # *float32, output BCx (B, 3H, L) contiguous
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,   # sizes
    stride_x_b, stride_x_l, stride_x_h,        # strides for x
    stride_w_j, stride_w_h, stride_w_l,        # strides for w
    stride_BCj, stride_BC_b, stride_BC_l       # strides for BCx
):
    # Grid: (3H_tiles, L_tiles, B)
    J = 3 * H
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)        # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128)      # [128]

    mask_j = j_offsets < J
    mask_l = l_offsets < L
    mask = mask_j[:, None] & mask_l[None, :]

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Accumulate BCx[b, j, l] = sum_h sum_l x[b, l, h] * w[j, h, l] + bias[j]
    for ho in range(0, H):
        # Load x[b, l, ho] for all l
        x_ptrs = x_ptr + b * stride_x_b + l_offsets[None, :] * stride_x_l + ho * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast along j

        # Load w[j, ho, l] for j_offsets and l_offsets
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + ho * stride_w_h + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        acc += x_vals * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store BCx[b, j, l]
    BCx_ptrs = BCx_ptr + b * stride_BC_b + j_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    tl.store(BCx_ptrs, acc, mask=mask)


# 2) Split BCx (B, 3H, L) into three tensors (B, L, H): B_tensor, C_tensor, x_proj
# We will implement chunking via copying slices along the feature dimension.
@triton.jit
def chunk3_kernel(
    BCx_ptr,            # *float32, input BCx (B, 3H, L) contiguous
    B_ptr,              # *float32, output B_tensor (B, L, H) contiguous
    C_ptr,              # *float32, output C_tensor (B, L, H) contiguous
    X_ptr,              # *float32, output x_proj (B, L, H) contiguous
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    stride_BC_b, stride_BC_j, stride_BC_l,
    stride_B_b, stride_B_l, stride_B_h,
    stride_C_b, stride_C_l, stride_C_h,
    stride_X_b, stride_X_l, stride_X_h,
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

    # B_tensor: BCx[:, h, :] for h in [0..H-1]
    BCx_ptrs_B = BCx_ptr + b * stride_BC_b + h_offsets[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_B = tl.load(BCx_ptrs_B, mask=mask, other=0.0)
    B_ptrs = B_ptr + b * stride_B_b + l_offsets[None, :] * stride_B_l + h_offsets[:, None] * stride_B_h
    tl.store(B_ptrs, vals_B, mask=mask)

    # C_tensor: BCx[:, H + h, :] for h in [0..H-1]
    BCx_ptrs_C = BCx_ptr + b * stride_BC_b + (H + h_offsets)[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_C = tl.load(BCx_ptrs_C, mask=mask, other=0.0)
    C_ptrs = C_ptr + b * stride_C_b + l_offsets[None, :] * stride_C_l + h_offsets[:, None] * stride_C_h
    tl.store(C_ptrs, vals_C, mask=mask)

    # x_proj: BCx[:, 2H + h, :] for h in [0..H-1]
    BCx_ptrs_X = BCx_ptr + b * stride_BC_b + (2 * H + h_offsets)[:, None] * stride_BC_j + l_offsets[None, :] * stride_BC_l
    vals_X = tl.load(BCx_ptrs_X, mask=mask, other=0.0)
    X_ptrs = X_ptr + b * stride_X_b + l_offsets[None, :] * stride_X_l + h_offsets[:, None] * stride_X_h
    tl.store(X_ptrs, vals_X, mask=mask)


# 3) Element-wise gating: Bx = B_tensor * x_proj
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


# 4) Grouped causal 1D convolution with kernel_size=4, stride=1, groups=H:
# Input Bx: (B, H, L), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
# We pad on the last dimension by K-1 = 3 (causal) so length becomes L + 3 for the conv.
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

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # For each output channel h and each position l, accumulate over k in [0..3]
    for ho in range(0, H):
        # Input y = Bx[:, h, :] with shape (L,)
        # We read y[b, l, ho] for l in l_offsets
        y_ptrs = Bx_ptr + b * stride_Bx_b + ho * stride_Bx_h + l_offsets[None, :] * stride_Bx_l
        # Mask valid l; causal masking handled by bounds checking within L
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast along ho later

        # Weight w[ho, ho, k] for k in [0..3]
        # Note: groups=H implies output channel index equals input channel index for each tap.
        for k in range(0, 4):
            w_ptrs = w_ptr + ho * stride_w_ho + ho * stride_w_hi + k * stride_w_k
            w_val = tl.load(w_ptrs)  # scalar
            # Apply causal: if l + k >= L, skip (since we only store up to l < L)
            acc += w_val * y_vals

    # Add bias
    b_vals = tl.load(bias_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store to out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask)


# 5) Out-projection: out = F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, L, H), out_proj_weight: (H, L, H), out_proj_bias: (H), out: (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,               # *float32, input y (B, L, H)
    w_ptr,               # *float32, weight (H, L, H)
    b_ptr,               # *float32, bias (H)
    out_ptr,             # *float32, output (B, L, H)
    B, L, H,
    stride_y_b, stride_y_l, stride_y_h,
    stride_w_ho, stride_w_l, stride_w_hi,
    stride_out_b, stride_out_l, stride_out_h,
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

    # For each output channel h: out[b, l, h] = sum_h sum_l y[b, l, h] * w[h, l, h] + bias[h]
    # Here, since out_proj_weight has shape (H, L, H), the output h is the channel we write.
    # We must sum over weight's L dimension and input h dimension. However, given y has shape (B, L, H),
    # the linear operation here is out[b, l, h] = sum over all h_in (H dimension of y), of y[b, l, h_in] * w[h_in, l, h] + bias[h].
    # That is, for each h, output across l and h_in:
    # but since out has shape (B, L, H), the correct understanding is that we compute a dot over y's H dimension for each (b, l, h).
    # However, PyTorch's F.linear(y, w, b) with w (H, L, H) computes out[b, l, h] = sum over h_in (dimension 0 of w) and l_in (dimension 1) or some reduction.
    # In our original model, out_proj_weight has shape (H, L, H) and F.linear(y, out_proj_weight, out_proj_bias) produces (B, L, H).
    # So the correct Triton formula is:
    # out[b, l, h] = sum over h_in in [0..H-1] of y[b, l, h_in] * w[h_in, l, h] + bias[h].
    for h_in in range(0, H):
        y_ptrs = y_ptr + b * stride_y_b + l_offsets[None, :] * stride_y_l + h_in * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_l[None, :], other=0.0)  # (1, 128), broadcast along h_offsets

        # w[h_in, l_offsets, h]
        w_ptrs = w_ptr + h_in * stride_w_ho + l_offsets[None, :] * stride_w_l + h_offsets[:, None] * stride_w_hi
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)  # (64, 128)

        acc += y_vals * w_vals  # broadcast y_vals along h_offsets

    # Add bias
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # (64,)
    acc += b_vals[:, None]

    # Store out[b, h, l]
    out_ptrs = out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
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
        # Ensure CUDA tensors and float32 for stable behavior
        assert x.is_cuda, "Input x must be on CUDA."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "All parameters must be on CUDA."
        B, L, H = x.shape
        device = x.device
        dtype = torch.float32

        # 1) Compute BCx: in-projection F.linear(x, in_proj_weight, in_proj_bias)
        x_contig = x.contiguous().to(dtype)
        w_contig = in_proj_weight.contiguous().to(dtype)
        b_in_contig = in_proj_bias.contiguous().to(dtype)

        BCx = torch.empty((B, 3 * H, L), device=device, dtype=dtype)

        J = 3 * H
        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x_contig, w_contig, b_in_contig, BCx,
            B=B, L=L, H=H,
            stride_x_b=x_contig.stride(0), stride_x_l=x_contig.stride(1), stride_x_h=x_contig.stride(2),
            stride_w_j=w_contig.stride(0), stride_w_h=w_contig.stride(1), stride_w_l=w_contig.stride(2),
            stride_BCj=BCx.stride(1), stride_BC_b=BCx.stride(0), stride_BC_l=BCx.stride(2),
        )

        # 2) Split BCx into B_tensor, C_tensor, x_proj (each (B, L, H)) using chunk3_kernel
        B_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        C_tensor = torch.empty((B, L, H), device=device, dtype=dtype)
        x_proj = torch.empty((B, L, H), device=device, dtype=dtype)

        BCx_contig = BCx.contiguous()
        grid_chunk = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        chunk3_kernel[grid_chunk](
            BCx_contig, B_tensor, C_tensor, x_proj,
            B=B, L=L, H=H,
            stride_BC_b=BCx_contig.stride(0), stride_BC_j=BCx_contig.stride(1), stride_BC_l=BCx_contig.stride(2),
            stride_B_b=B_tensor.stride(0), stride_B_l=B_tensor.stride(1), stride_B_h=B_tensor.stride(2),
            stride_C_b=C_tensor.stride(0), stride_C_l=C_tensor.stride(1), stride_C_h=C_tensor.stride(2),
            stride_X_b=x_proj.stride(0), stride_X_l=x_proj.stride(1), stride_X_h=x_proj.stride(2),
        )

        # 3) Element-wise gating: Bx = B_tensor * x_proj
        Bx = torch.empty((B, H, L), device=device, dtype=dtype)
        grid_gate = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_gate](
            B_tensor, x_proj, Bx,
            B=B, L=L, H=H,
            stride_a_b=B_tensor.stride(0), stride_a_l=B_tensor.stride(1), stride_a_h=B_tensor.stride(2),
            stride_b_b=x_proj.stride(0), stride_b_l=x_proj.stride(1), stride_b_h=x_proj.stride(2),
            stride_out_b=Bx.stride(0), stride_out_l=Bx.stride(2), stride_out_h=Bx.stride(1),
        )

        # 4) Grouped causal 1D conv: conv_out (B, H, L)
        conv_weight_contig = conv_weight.contiguous().to(dtype)  # (H, H, 4)
        conv_bias_contig = conv_bias.contiguous().to(dtype)      # (H)

        conv_out = torch.empty((B, H, L), device=device, dtype=dtype)
        grid_conv = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight_contig, conv_bias_contig, conv_out,
            B=B, L=L, H=H,
            stride_Bx_b=Bx.stride(0), stride_Bx_h=Bx.stride(1), stride_Bx_l=Bx.stride(2),
            stride_w_ho=conv_weight_contig.stride(0), stride_w_hi=conv_weight_contig.stride(1), stride_w_k=conv_weight_contig.stride(2),
            stride_out_b=conv_out.stride(0), stride_out_h=conv_out.stride(1), stride_out_l=conv_out.stride(2),
        )

        # 5) Output gating and projection: y = C * conv_out; then out = linear(y, out_proj_weight, out_proj_bias)
        # Implement y via Triton multiply: y (B, L, H)
        y = torch.empty((B, L, H), device=device, dtype=dtype)
        grid_mul = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        gate_mul_kernel[grid_mul](
            C_tensor, conv_out, y,
            B=B, L=L, H=H,
            stride_a_b=C_tensor.stride(0), stride_a_l=C_tensor.stride(1), stride_a_h=C_tensor.stride(2),
            stride_b_b=conv_out.stride(0), stride_b_l=conv_out.stride(2), stride_b_h=conv_out.stride(1),
            stride_out_b=y.stride(0), stride_out_l=y.stride(1), stride_out_h=y.stride(2),
        )

        # 6) Out-projection: F.linear(y, out_proj_weight, out_proj_bias) to (B, L, H)
        out_proj_weight_contig = out_proj_weight.contiguous().to(dtype)  # (H, L, H)
        out_proj_bias_contig = out_proj_bias.contiguous().to(dtype)      # (H)
        output = torch.empty((B, L, H), device=device, dtype=dtype)

        grid_out = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        out_proj_kernel[grid_out](
            y, out_proj_weight_contig, out_proj_bias_contig, output,
            B=B, L=L, H=H,
            stride_y_b=y.stride(0), stride_y_l=y.stride(1), stride_y_h=y.stride(2),
            stride_w_ho=out_proj_weight_contig.stride(0), stride_w_l=out_proj_weight_contig.stride(1), stride_w_hi=out_proj_weight_contig.stride(2),
            stride_out_b=output.stride(0), stride_out_l=output.stride(1), stride_out_h=output.stride(2),
        )

        return output


def run(*args):
    return ModelNew()(*args)
