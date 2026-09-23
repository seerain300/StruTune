import torch
import torch.nn as nn
import triton
import triton.language as tl

# 1) In-projection: compute BCx[b, j, l] = sum_k in_proj_weight[j, k, l] * x[b, k, l] + bias[j]
# Shapes:
#   x: (B, L, H) contiguous
#   in_proj_weight: (3H, H, L) contiguous
#   in_proj_bias: (3H)
# Output:
#   out: (J, L) where J=3*H; we will view as (B, J, L) after the kernel
@triton.jit
def in_proj_kernel_B(
    x_ptr,                 # *float32, base pointer to x (B, L, H)
    w_ptr,                 # *float32, base pointer to in_proj_weight (3H, H, L)
    b_ptr,                 # *float32, base pointer to in_proj_bias (3H)
    out_ptr,               # *float32, base pointer to output (J, L)
    B, L, H, J,            # int32 sizes: B=batch, L=seq_len, H=hidden, J=3*H
    stride_x_b, stride_x_l, stride_x_h,   # strides for x
    stride_w_j, stride_w_k, stride_w_l,   # strides for in_proj_weight
    stride_out_j, stride_out_l            # strides for output
):
    # Grid: (J_tiles, L_tiles, B)
    j_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)  # batch index

    j_offsets = j_block * 64 + tl.arange(0, 64)   # [64]
    l_offsets = l_block * 128 + tl.arange(0, 128) # [128]
    mask_j = j_offsets < J
    mask_l = l_offsets < L

    acc = tl.zeros((64, 128), dtype=tl.float32)

    # Loop over k = 0..H-1
    for k in range(0, H):
        # Load x[b, k, l] across l_offsets
        x_vals = tl.load(
            x_ptr + b * stride_x_b + l_offsets * stride_x_l + k * stride_x_h,
            mask=mask_l, other=0.0
        )  # (128,)
        # Load in_proj_weight[j, k, l] across j_offsets and l_offsets
        w_vals = tl.load(
            w_ptr + j_offsets[:, None] * stride_w_j + k * stride_w_k + l_offsets[None, :] * stride_w_l,
            mask=mask_j[:, None] & mask_l[None, :], other=0.0
        )  # (64, 128)
        acc += x_vals[None, :] * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + j_offsets, mask=mask_j, other=0.0)  # (64,)
    acc += bias_vals[:, None]

    # Store output[j, l] for this batch b
    tl.store(out_ptr + j_offsets[:, None] * stride_out_j + l_offsets[None, :] * stride_out_l,
             acc, mask=mask_j[:, None] & mask_l[None, :])

# 2) Grouped causal 1D convolution with kernel_size=4, groups=H:
# Input Bx: (B, H, L)
# Weight: (H, H, 4)
# Output: (B, H, L)
# groups=H means each output channel c uses its own input channel c for conv along L.
@triton.jit
def conv1d_grouped_causal_kernel(
    Bx_ptr,                # *float32, base pointer to input (B, H, L)
    w_ptr,                 # *float32, base pointer to conv_weight (H, H, 4)
    bias_ptr,              # *float32, base pointer to conv_bias (H)
    out_ptr,               # *float32, base pointer to output (B, H, L)
    B, L, H,               # int32 sizes
    stride_bx_b, stride_bx_c, stride_bx_l,   # strides for Bx
    stride_w_o, stride_w_i, stride_w_k,      # strides for conv_weight
    stride_out_b, stride_out_c, stride_out_l,  # strides for output
    K: tl.constexpr             # kernel_size (4)
):
    # Grid: (B, H, ceil(L/128))
    b = tl.program_id(0)
    c = tl.program_id(1)
    l_block = tl.program_id(2)

    l_offsets = l_block * 128 + tl.arange(0, 128)
    mask_l = l_offsets < L

    acc = tl.zeros((128,), dtype=tl.float32)

    # For each k in [0..K-1], accumulate Bx[b, c, l + k] * w[c, c, k] and add bias[c]
    for k in range(0, K):
        # Load Bx[b, c, l + k]
        bx_vals = tl.load(
            Bx_ptr + b * stride_bx_b + c * stride_bx_c + (l_offsets + k) * stride_bx_l,
            mask=mask_l, other=0.0
        )  # (128,)
        # Load conv_weight[c, c, k] (scalar per k)
        w_val = tl.load(w_ptr + c * stride_w_o + c * stride_w_i + k * stride_w_k)
        acc += bx_vals * w_val

    # Add bias
    b_bias = tl.load(bias_ptr + c)
    acc += b_bias

    # Store output[b, c, l]
    tl.store(out_ptr + b * stride_out_b + c * stride_out_c + l_offsets * stride_out_l,
             acc, mask=mask_l)

# 3) Out-projection: computes y (B, L, H) -> output (B, L, H)
# weight shape (H, L, H), bias (H)
@triton.jit
def out_proj_kernel_b(
    y_ptr,             # *float32, base pointer to y (B, L, H)
    wout_ptr,          # *float32, base pointer to out_proj_weight (H, L, H)
    bout_ptr,          # *float32, base pointer to out_proj_bias (H)
    out_ptr,           # *float32, base pointer to out (B, L, H)
    B, L, H,           # int32 sizes
    stride_y_b, stride_y_l, stride_y_h,   # strides for y
    stride_w_h, stride_w_l, stride_w_k,   # strides for out_proj_weight
    stride_out_b, stride_out_l, stride_out_h, # strides for output
    BLOCK_M: tl.constexpr,               # tile over H (channels)
    BLOCK_N: tl.constexpr,               # tile over L (time)
):
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)  # batch index

    h_offsets = h_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    l_offsets = l_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_h = h_offsets < H
    mask_l = l_offsets < L

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # We compute output[b, h, l] = sum_k y[b, l, k] * out_proj_weight[h, l, k] + bias[h]
    for k in range(0, H):
        y_vals = tl.load(
            y_ptr + b * stride_y_b + l_offsets * stride_y_l + k * stride_y_h,
            mask=mask_l, other=0.0
        )  # (BLOCK_N,)
        wout_vals = tl.load(
            wout_ptr + h_offsets[:, None] * stride_w_h + l_offsets[None, :] * stride_w_l + k * stride_w_k,
            mask=mask_h[:, None] & mask_l[None, :], other=0.0
        )  # (BLOCK_M, BLOCK_N)
        acc += y_vals[None, :] * wout_vals

    # Add bias
    bias_vals = tl.load(bout_ptr + h_offsets, mask=mask_h, other=0.0)  # (BLOCK_M,)
    acc += bias_vals[:, None]

    # Store output[b, h, l]
    tl.store(out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l,
             acc, mask=mask_h[:, None] & mask_l[None, :])

# 4) Elementwise gate multiply kernel: out = a * b
@triton.jit
def gate_mul_kernel(
    a_ptr, b_ptr, out_ptr,
    B, L, H,
    stride_a_b, stride_a_l, stride_a_h,
    stride_b_b, stride_b_l, stride_b_h,
    stride_out_b, stride_out_l, stride_out_h,
    BLOCK_M: tl.constexpr,   # tile over H
    BLOCK_N: tl.constexpr,   # tile over L
):
    h_block = tl.program_id(0)
    l_block = tl.program_id(1)
    b = tl.program_id(2)  # batch index
    h_offsets = h_block * BLOCK_M + tl.arange(0, BLOCK_M)
    l_offsets = l_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_h = h_offsets < H
    mask_l = l_offsets < L

    a_vals = tl.load(a_ptr + b * stride_a_b + l_offsets * stride_a_l + h_offsets * stride_a_h,
                     mask=mask_h[:, None] & mask_l[None, :], other=0.0)
    b_vals = tl.load(b_ptr + b * stride_b_b + l_offsets * stride_b_l + h_offsets * stride_b_h,
                     mask=mask_h[:, None] & mask_l[None, :], other=0.0)
    out_vals = a_vals * b_vals
    tl.store(out_ptr + b * stride_out_b + h_offsets[:, None] * stride_out_h + l_offsets[None, :] * stride_out_l,
             out_vals, mask=mask_h[:, None] & mask_l[None, :])

# Host-side functions that launch Triton kernels
def in_proj_triton(x, in_proj_weight, in_proj_bias):
    B, L, H = x.shape
    J = 3 * H
    # Ensure contiguous and float32
    x_c = x.contiguous().float()
    w_c = in_proj_weight.contiguous().float()
    b_c = in_proj_bias.contiguous().float()
    # Output buffer (J, L)
    out = torch.empty((J, L), dtype=torch.float32, device=x.device)
    stride_x_b, stride_x_l, stride_x_h = x_c.stride()
    stride_w_j, stride_w_k, stride_w_l = w_c.stride()
    stride_out_j, stride_out_l = out.stride()
    # Launch grid: (J_tiles, L_tiles, B)
    grid = (triton.cdiv(J, 64), triton.cdiv(L, 128), B)
    in_proj_kernel_B[grid](
        x_c, w_c, b_c, out,
        B, L, H, J,
        stride_x_b, stride_x_l, stride_x_h,
        stride_w_j, stride_w_k, stride_w_l,
        stride_out_j, stride_out_l,
        num_warps=4, num_stages=2
    )
    return out.view(B, J, L)  # (B, 3H, L)

def conv1d_grouped_causal_triton(Bx, conv_weight, conv_bias):
    # Bx: (B, H, L), conv_weight: (H, H, 4), conv_bias: (H)
    B, H, L = Bx.shape
    Bx_c = Bx.contiguous().float()
    w_c = conv_weight.contiguous().float()
    b_bias = conv_bias.contiguous().float()
    out = torch.empty((B, H, L), dtype=torch.float32, device=Bx.device)
    # Strides for Bx
    stride_bx_b, stride_bx_c, stride_bx_l = Bx_c.stride()
    stride_w_o, stride_w_i, stride_w_k = w_c.stride()
    stride_out_b, stride_out_c, stride_out_l = out.stride()
    grid = (B, H, triton.cdiv(L, 128))
    conv1d_grouped_causal_kernel[grid](
        Bx_c, w_c, b_bias, out,
        B, L, H,
        stride_bx_b, stride_bx_c, stride_bx_l,
        stride_w_o, stride_w_i, stride_w_k,
        stride_out_b, stride_out_c, stride_out_l,
        K=4,
        num_warps=4, num_stages=2
    )
    return out

def out_proj_triton(y, out_proj_weight, out_proj_bias):
    B, L, H = y.shape
    y_c = y.contiguous().float()
    wout_c = out_proj_weight.contiguous().float()
    bout_c = out_proj_bias.contiguous().float()
    out = torch.empty((B, L, H), dtype=torch.float32, device=y.device)
    stride_y_b, stride_y_l, stride_y_h = y_c.stride()
    stride_w_h, stride_w_l, stride_w_k = wout_c.stride()
    stride_out_b, stride_out_l, stride_out_h = out.stride()
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N), B)
    out_proj_kernel_b[grid](
        y_c, wout_c, bout_c, out,
        B, L, H,
        stride_y_b, stride_y_l, stride_y_h,
        stride_w_h, stride_w_l, stride_w_k,
        stride_out_b, stride_out_l, stride_out_h,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )
    return out

def gate_mul_triton(a, b):
    B, L, H = a.shape
    out = torch.empty((B, L, H), dtype=torch.float32, device=a.device)
    stride_a_b, stride_a_l, stride_a_h = a.stride()
    stride_b_b, stride_b_l, stride_b_h = b.stride()
    stride_out_b, stride_out_l, stride_out_h = out.stride()
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N), B)
    gate_mul_kernel[grid](
        a, b, out,
        B, L, H,
        stride_a_b, stride_a_l, stride_a_h,
        stride_b_b, stride_b_l, stride_b_h,
        stride_out_b, stride_out_l, stride_out_h,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )
    return out

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
        # Cast to float32 for Triton kernels
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        # Step 1: In-projection
        # Input x: (B, L,


def run(*args):
    return ModelNew()(*args)
