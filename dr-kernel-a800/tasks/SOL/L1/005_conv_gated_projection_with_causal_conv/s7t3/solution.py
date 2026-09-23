import torch
import torch.nn as nn
import triton
import triton.language as tl

# Kernel 1: In-projection linear for x -> (B, 3H, L)
# Input x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# Output BCx: (B, 3H, L). We produce out_ptr of shape (B, J, L) directly, then view (B, 3H, L) in host.
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *f32, base ptr to x (B, L, H)
    w_ptr,                 # *f32, base ptr to in_proj_weight (3H, H, L)
    b_ptr,                 # *f32, base ptr to in_proj_bias (3H)
    out_ptr,               # *f32, base ptr to output (B, J, L)
    B, L, H, J,            # sizes: J = 3*H
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_out_b, stride_out_j, stride_out_l  # strides for output (B, J, L)
):
    # Grid dims: (J_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    j_offsets = j_block * 64 + tl.arange(0, 64)   # tile over J=3H
    l_offsets = l_block * 128 + tl.arange(0, 128) # tile over L

    mask_j = j_offsets < J
    mask_l = l_offsets < L

    # Accumulator for (J, L) tile
    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over input feature k (0..H-1)
    for k in range(0, H):
        # Load x[b, l, k] across l_offsets
        x_ptrs = x_ptr + b * stride_x_b + l_offsets * stride_x_l + k * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l, other=0.0)  # [128]

        # Load in_proj_weight[j, k, l] across j_offsets and l_offsets
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + k * stride_w_k + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask_j[:, None] & mask_l[None, :], other=0.0)  # [64, 128]

        # Outer product accumulate
        acc += w_vals * x_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # [64]
    acc += b_vals[:, None]

    # Store to output at [b, j, l]
    out_ptrs = out_ptr + b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask_j[:, None] & mask_l[None, :])


# Kernel 2: Grouped causal 1D convolution on Bx (shape: (B, H, L)) with kernel_size=4, stride=1, groups=H.
# Input Bx: (B, H, L), conv_weight: (H, H, 4), conv_bias: (H)
# Output conv_out: (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *f32, base ptr to Bx (B, H, L)
    w_ptr,                 # *f32, base ptr to conv_weight (H, H, 4)
    b_bias_ptr,            # *f32, base ptr to conv_bias (H)
    out_ptr,               # *f32, base ptr to output (B, H, L)
    B, L, H, K,            # sizes, K=4
    stride_bx_b, stride_bx_c, stride_bx_l,   # strides for Bx
    stride_w_o, stride_w_i, stride_w_k,      # strides for conv_weight
    stride_out_b, stride_out_c, stride_out_l  # strides for output
):
    # Grid dims: (H, L_tiles, B)
    c = tl.program_id(0)   # output channel index (0..H-1)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128], tile along L
    mask_l = l_offsets < L

    # Initialize accumulator for this (b, c) and L tile
    acc = tl.zeros((128,), dtype=tl.float32)

    # Causal conv with kernel_size=4
    for k_idx in range(0, K):
        l_in = l_offsets - (k_idx - 0)
        valid = (l_in >= 0) & (l_in < L) & mask_l
        # Load Bx[b, c, l_in]
        bx_ptrs = Bx_ptr + b * stride_bx_b + c * stride_bx_c + l_in * stride_bx_l
        bx_vals = tl.load(bx_ptrs, mask=valid, other=0.0)  # [128]
        # Load conv_weight[c, c, k_idx] scalar
        w_val = tl.load(w_ptr + c * stride_w_o + c * stride_w_i + k_idx * stride_w_k)
        acc += bx_vals * w_val

    # Add bias
    bias = tl.load(b_bias_ptr + c)
    acc += bias

    # Store output conv_out[b, c, l]
    out_ptrs = out_ptr + b * stride_out_b + c * stride_out_c + l_offsets * stride_out_l
    tl.store(out_ptrs, acc, mask=mask_l)


# Kernel 3: Out-projection linear for y -> (B, L, H)
# Input y: (B, L, H), out_proj_weight: (H, L, H), out_proj_bias: (H)
# Output out: (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,                 # *f32, base ptr to y (B, L, H)
    w_ptr,                 # *f32, base ptr to out_proj_weight (H, L, H)
    b_ptr,                 # *f32, base ptr to out_proj_bias (H)
    out_ptr,               # *f32, base ptr to output (B, L, H)
    B, L, H,               # sizes
    stride_y_b, stride_y_l, stride_y_h,   # strides for y
    stride_w_h, stride_w_l, stride_w_k,   # strides for out_proj_weight (H, L, H)
    stride_out_b, stride_out_l, stride_out_h  # strides for output
):
    # Grid dims: (H_tiles, L_tiles, B)
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)

    h = h_block
    l_offsets = l_block * 128 + tl.arange(0, 128)  # [128], tile over L
    mask_l = l_offsets < L

    # Accumulator for output vector of length L tile
    acc = tl.zeros((128,), dtype=tl.float32)

    # Loop over input dimension K = L
    for k in range(0, L):
        # Load y[b, k, h]
        y_val = tl.load(y_ptr + b * stride_y_b + k * stride_y_l + h * stride_y_h)
        # Load out_proj_weight[h, k, h] across l_offsets
        w_ptrs = w_ptr + h * stride_w_h + k * stride_w_l + l_offsets * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_l, other=0.0)
        acc += y_val * w_vals  # broadcast scalar

    # Add bias
    bias = tl.load(b_ptr + h)
    acc += bias

    # Store output out[b, l, h]
    out_ptrs = out_ptr + b * stride_out_b + l_offsets * stride_out_l + h * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_l)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure contiguous and float32 for Triton
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        B, L, H = x.shape
        J = 3 * H

        # Step 1: In-projection to get BCx of shape (B, 3H, L)
        # Output buffer: (B, J, L)
        BCx_tmp = torch.empty((B, J, L), dtype=torch.float32, device=x.device)

        # Strides
        stride_x_b, stride_x_l, stride_x_h = x.stride()
        stride_w_j, stride_w_k, stride_w_l = in_proj_weight.stride()
        stride_out_b, stride_out_j, stride_out_l = BCx_tmp.stride()

        # Launch in_proj_kernel_B
        grid_in = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx_tmp,
            B, L, H, J,
            stride_x_b, stride_x_l, stride_x_h,
            stride_w_j, stride_w_k, stride_w_l,
            stride_out_b, stride_out_j, stride_out_l,
            num_warps=4, num_stages=2
        )

        # Reshape to (B, 3H, L) and transpose to (B, L, 3H) for chunking
        BCx = BCx_tmp.view(B, J, L)
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, L, 3H)

        # Chunk 3 along last dim: B, C, x_proj
        B_tensor, C_tensor, x_proj = torch.chunk(BCx_T, 3, dim=1)

        # Step 2: Element-wise gating Bx = B * x_proj (trivial elementwise op in PyTorch)
        Bx = B_tensor * x_proj  # (B, L, H)
        Bx = Bx.contiguous()

        # Step 3: Grouped causal 1D conv with kernel_size=4, stride=1, groups=H
        # Input Bx: (B, H, L), conv_weight: (H, H, 4), conv


def run(*args):
    return ModelNew()(*args)
