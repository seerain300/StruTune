import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for in-projection: computes y = x @ in_proj_weight^T + in_proj_bias
# Input x: (B, L, H), in_proj_weight: (3H, H, L), in_proj_bias: (3H)
# Output y: (B, 3H, L) with strides given
@triton.jit
def in_proj_kernel(
    x_ptr,                 # *float32, x: (B, L, H)
    w_ptr,                 # *float32, in_proj_weight: (3H, H, L)
    b_ptr,                 # *float32, in_proj_bias: (3H)
    out_ptr,               # *float32, out: (B, 3H, L)
    B, L, H,               # sizes
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_out_b, stride_out_j, stride_out_l,  # strides for out
    BLOCK_J: tl.constexpr,                 # tile size for J=3H
    BLOCK_L: tl.constexpr                  # tile size for L
):
    pid_b = tl.program_id(0)  # batch
    pid_j = tl.program_id(1)  # feature tile over 3H
    pid_l = tl.program_id(2)  # sequence tile over L

    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # [BLOCK_J]
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]

    mask_j = j_offsets < (3 * H)
    mask_l = l_offsets < L

    acc = tl.zeros((BLOCK_J, BLOCK_L), dtype=tl.float32)

    # Loop over input features k in [0, H)
    for k in range(0, H):
        # Weight w[j, k, l] for all j in tile and l in tile
        w_ptrs = w_ptr + j_offsets[:, None] * stride_w_j + k * stride_w_k + l_offsets[None, :] * stride_w_l
        w_vals = tl.load(w_ptrs, mask=mask_j[:, None] & mask_l[None, :], other=0.0)

        # x[b, l, k] for all l in tile
        x_ptrs = x_ptr + pid_b * stride_x_b + l_offsets[None, :] * stride_x_l + k * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_l[None, :], other=0.0)  # [1, BLOCK_L]

        # Outer product accumulate
        acc += w_vals * x_vals  # [BLOCK_J, BLOCK_L]

    # Add bias per j
    b_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # [BLOCK_J]
    acc += b_vals[:, None]

    # Store to out[b, j, l]
    out_ptrs = out_ptr + pid_b * stride_out_b + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask_j[:, None] & mask_l[None, :])


# Triton kernel for grouped causal 1D convolution: F.conv1d(Bx, conv_weight, conv_bias, groups=H)
# Input Bx: (B, H, L), conv_weight: (H, H, 4), conv_bias: (H)
# Output: conv_out (B, H, L)
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, Bx: (B, H, L)
    w_ptr,                 # *float32, conv_weight: (H, H, 4)
    b_ptr,                 # *float32, conv_bias: (H)
    out_ptr,               # *float32, conv_out: (B, H, L)
    B, L, H,               # sizes
    stride_bx_b, stride_bx_h, stride_bx_l,  # strides for Bx
    stride_w_o, stride_w_i, stride_w_k,     # strides for conv_weight
    stride_out_b, stride_out_h, stride_out_l,  # strides for out
    K: tl.constexpr,                     # kernel_size (4)
    BLOCK_H: tl.constexpr,               # tile over H
    BLOCK_L: tl.constexpr                # tile over L
):
    pid_b = tl.program_id(0)   # batch
    pid_h = tl.program_id(1)   # feature channel tile
    pid_l = tl.program_id(2)   # sequence tile

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)     # [BLOCK_H]
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)     # [BLOCK_L]

    mask_h = h_offsets < H
    mask_l = l_offsets < L

    acc = tl.zeros((BLOCK_H, BLOCK_L), dtype=tl.float32)

    # For grouped conv with groups=H, each output channel h uses its own input channel h
    for c in range(0, H):
        for kk in range(0, K):
            # Causal input index: t = l_offsets - kk
            t = l_offsets - kk
            mask_t = t >= 0
            # Load Bx[pid_b, c, t]
            bx_ptrs = Bx_ptr + pid_b * stride_bx_b + c * stride_bx_h + t * stride_bx_l
            bx_vals = tl.load(bx_ptrs, mask=mask_l & mask_t, other=0.0)  # [BLOCK_L]

            # Load conv_weight[c, c, kk]
            w_val = tl.load(w_ptr + c * stride_w_o + c * stride_w_i + kk * stride_w_k)  # scalar

            # Accumulate
            acc += w_val * bx_vals[None, :]

    # Add bias per output channel
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc += b_vals[:, None]

    # Store conv_out[pid_b, h_offsets, l_offsets]
    out_ptrs = out_ptr + pid_b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l
    tl.store(out_ptrs, acc, mask=mask_h[:, None] & mask_l[None, :])


# Triton kernel for out-projection: computes y @ out_proj_weight^T + out_proj_bias
# Input y: (B, L, H), out_proj_weight: (H, L, H), out_proj_bias: (H)
# Output: (B, L, H)
@triton.jit
def out_proj_kernel(
    y_ptr,                 # *float32, y: (B, L, H)
    w_ptr,                 # *float32, out_proj_weight: (H, L, H)
    b_ptr,                 # *float32, out_proj_bias: (H)
    out_ptr,               # *float32, out: (B, L, H)
    B, L, H,               # sizes
    stride_y_b, stride_y_l, stride_y_h,       # strides for y
    stride_w_h, stride_w_l, stride_w_k,       # strides for out_proj_weight
    stride_out_b, stride_out_l, stride_out_h, # strides for out
    BLOCK_H: tl.constexpr,                    # tile over H
    BLOCK_L: tl.constexpr                     # tile over L
):
    pid_b = tl.program_id(0)   # batch
    pid_h = tl.program_id(1)   # output channel tile
    pid_l = tl.program_id(2)   # sequence tile

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)     # [BLOCK_H]
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)     # [BLOCK_L]

    mask_h = h_offsets < H
    mask_l = l_offsets < L

    acc = tl.zeros((BLOCK_H, BLOCK_L), dtype=tl.float32)

    # Loop over input K dimension (L)
    for k in range(0, L):
        # Load y[pid_b, k, h_offsets]
        y_ptrs = y_ptr + pid_b * stride_y_b + k * stride_y_l + h_offsets * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_h, other=0.0)  # [BLOCK_H]

        # Load out_proj_weight[h_offsets, k, l_offsets]
        w_ptrs = w_ptr + h_offsets[:, None] * stride_w_h + k * stride_w_l + l_offsets[None, :] * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_h[:, None] & mask_l[None, :], other=0.0)

        acc += y_vals[:, None] * w_vals  # [BLOCK_H, BLOCK_L]

    # Add bias per output channel
    b_vals = tl.load(b_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc += b_vals[:, None]

    # Store out[pid_b, l_offsets, h_offsets] (we write (B, L, H) with transposed indices)
    out_ptrs = out_ptr + pid_b * stride_out_b + l_offsets[None, :] * stride_out_l + h_offsets[:, None] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_l[None, :] & mask_h[:, None])


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
        # Ensure contiguous and float32 for Triton kernels
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        B, L, H = x.shape
        J = 3 * H

        # 1) In-projection: BCx of shape (B, 3H, L)
        BCx = torch.empty((B, J, L), dtype=torch.float32, device=x.device)

        # Strides
        stride_x_b, stride_x_l, stride_x_h = x.stride()
        stride_w_j, stride_w_k, stride_w_l = in_proj_weight.stride()
        stride_out_b, stride_out_j, stride_out_l = BCx.stride()

        # Launch in-projection kernel
        BLOCK_J = 64
        BLOCK_L = 128
        grid_in = (B, triton.cdiv(J, BLOCK_J), triton.cdiv(L, BLOCK_L))
        in_proj_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            stride_x_b, stride_x_l, stride_x_h,
            stride_w_j, stride_w_k, stride_w_l,
            stride_out_b, stride_out_j, stride_out_l,
            BLOCK_J=BLOCK_J, BLOCK_L=BLOCK_L,
            num_warps=4, num_stages=2
        )

        # Reshape/transposes to match original layout: (B, L, 3H)
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, L, 3H)
        B_tensor, C_tensor, x_proj = torch.chunk(BCx_T, 3, dim=1)  # each (B, L, H)

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = B_tensor * x_proj  # (B, L, H)
        Bx = Bx.contiguous()

        # 3) Grouped causal 1D conv with kernel_size=4, groups=H
        conv_out = torch.empty((B, H, L), dtype=torch.float32, device=x.device)

        stride_bx_b, stride_bx_h, stride_bx_l = Bx.stride()
        stride_w_o, stride_w_i, stride_w_k = conv_weight.stride()
        stride_out_b, stride_out_h, stride_out_l = conv_out.stride()

        BLOCK_H = 64
        BLOCK_L = 128
        grid_conv = (B, triton.cdiv(H, BLOCK_H), triton.cdiv(L, BLOCK_L))
        conv1d_grouped_causal_kernel[grid_conv](
            Bx, conv_weight, conv_bias, conv_out,
            B, L, H,
            stride_bx_b, stride_bx_h, stride_bx_l,
            stride_w_o, stride_w_i, stride_w_k,
            stride_out_b, stride_out_h, stride_out_l,
            K=4,
            BLOCK_H=BLOCK_H, BLOCK_L=BLOCK_L,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C * conv_out
        y = C_tensor * conv_out  # (B, H, L)

        # Transpose to (B, L, H) for out-projection
        y_T = y.transpose(-1, -2).contiguous()  # (B, L, H)

        # 5) Final out-projection: (B, L, H)
        output = torch.empty((B, L, H), dtype=torch.float32, device=x.device)

        stride_y_b, stride_y_l, stride_y_h = y_T.stride()
        stride_w_h, stride_w_l, stride_w_k = out_proj_weight.stride()
        stride_out_b, stride_out_l, stride_out_h = output.stride()

        BLOCK_H_out = 64
        BLOCK_L_out = 128
        grid_out = (B, triton.cdiv(H, BLOCK_H_out), triton.cdiv(L, BLOCK_L_out))
        out_proj_kernel[grid_out](
            y_T, out_proj_weight, out_proj_bias, output,
            B, L, H,
            stride_y_b, stride_y_l, stride_y_h,
            stride_w_h, stride_w_l, stride_w_k,
            stride_out_b, stride_out_l, stride_out_h,
            BLOCK_H=BLOCK_H_out, BLOCK_L=BLOCK_L_out,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
