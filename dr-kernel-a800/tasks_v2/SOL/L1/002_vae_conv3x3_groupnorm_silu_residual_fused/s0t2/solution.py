import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: 2D convolution with 3x3, stride=1, padding=1, no bias.
# Input: x[B, C_in, H, W], weight[C_in, C_out, 3, 3], output[B, C_out, H, W]
# We pass weight as a flattened array of length (C_in * C_out * 9) and index accordingly.
# Each program computes a BLOCK_H x BLOCK_W tile for a given (n, co).
@triton.jit
def conv3x3_stride1_pad1_tiled_kernel(
    x_ptr,              # *const float
    w_ptr,              # *const float, flattened weights (length = C_in * C_out * 9)
    out_ptr,            # *float (same dtype as x)
    B, C_in, C_out, H, W, H_out, W_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_y = tl.program_id(2)
    pid_x = tl.program_id(3)

    n = pid_n
    co = pid_co

    # Tile coordinates
    ys = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    xs = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
    Y, X = tl.meshgrid(ys, xs)

    # Masks for boundaries
    mask = (Y < H_out) & (X < W_out)

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Reduction over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            h_in = Y + kh - 1  # padding=1
            for kw in range(0, 3):
                w_in = X + kw - 1  # padding=1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask
                x_idx = n * C_in * H * W + ci * H * W + h_in * W + w_in
                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)
                x_val = x_val.to(tl.float32)
                # Load corresponding weight: ((ci * C_out + co) * 9) + (kh * 3 + kw)
                w_idx = (ci * C_out + co) * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                w_val = w_val.to(tl.float32)
                acc += x_val * w_val

    # Store output: out[n, co, Y, X]
    out_idx = n * C_out * H_out * W_out + co * (H_out * W_out) + Y * W_out + X
    # Cast back to original dtype of out_ptr (same as x)
    tl.store(out_ptr + out_idx, acc, mask=mask)


# Triton kernel: GroupNorm over NCHW, num_groups as constexpr.
# Assumptions: C is divisible by num_groups. Two-pass: compute mean/var per (n, group),
# then write normalized outputs using gamma/beta. Operates in fp32, casts back.
@triton.jit
def groupnorm_kernel(
    input_ptr,          # *const float
    output_ptr,         # *float
    gamma_ptr,          # *const float, shape [C]
    beta_ptr,           # *const float, shape [C]
    B, C, H, W,         # int32
    group_size,         # int32 = C // num_groups
    num_groups: tl.constexpr,
    eps,                # float32
    N_INPUT_ELEMENTS,   # int32 = B*C*H*W
    BLOCK: tl.constexpr
):
    # Grid: one program per (n, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    base_c = g * group_size
    HW = H * W

    # First pass: compute sum and sum of squares for each channel in group
    sum_vec = tl.zeros([group_size], dtype=tl.float32)
    sumsq_vec = tl.zeros([group_size], dtype=tl.float32)

    for ci in range(0, group_size):
        c = base_c + ci
        s = 0.0
        ss = 0.0
        for i in range(0, HW):
            h = i // W
            w = i % W
            in_idx = n * C * HW + c * HW + h * W + w
            x = tl.load(input_ptr + in_idx)
            x32 = x.to(tl.float32)
            s += x32
            ss += x32 * x32
        sum_vec[ci] = s
        sumsq_vec[ci] = ss

    # Compute mean and var per channel
    M = float(HW * group_size)
    mean_vec = sum_vec / M
    var_vec = sumsq_vec / M - mean_vec * mean_vec
    inv_std_vec = tl.rsqrt(var_vec + eps)

    # Second pass: write normalized outputs
    for ci in range(0, group_size):
        c = base_c + ci
        for i in range(0, HW):
            h = i // W
            w = i % W
            in_idx = n * C * HW + c * HW + h * W + w
            x = tl.load(input_ptr + in_idx)
            x32 = x.to(tl.float32)
            y32 = (x32 - mean_vec[ci]) * inv_std_vec[ci]
            gamma = tl.load(gamma_ptr + c).to(tl.float32)
            beta = tl.load(beta_ptr + c).to(tl.float32)
            y32 = y32 * gamma + beta
            out_idx = n * C * HW + c * HW + h * W + w
            # We store as fp32; if output is fp32 input_ptr, this is fine.
            # Note: the host code ensures tensors are fp32 for these kernels.
            tl.store(output_ptr + out_idx, y32)


# Triton kernel: SiLU elementwise over 1D buffer
@triton.jit
def silu_kernel_1d(
    x_ptr,              # *const float
    out_ptr,            # *float
    size,               # int32 total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x32))
    y32 = x32 * sig
    # Store fp32 output
    tl.store(out_ptr + offs, y32, mask=mask)


# Triton kernel: elementwise add residual (out = out + x) over 1D buffer
@triton.jit
def add_residual_kernel_1d(
    out_ptr,            # *const float
    x_ptr,              # *const float
    size,               # int32 total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    res = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out32 = out.to(tl.float32)
    res32 = res.to(tl.float32)
    tl.store(out_ptr + offs, out32 + res32, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_size: int = 1024):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_size = block_size
        # Tile sizes for conv. You can tune these for performance.
        self.BLOCK_H = 8
        self.BLOCK_W = 8

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure CUDA tensors
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        B, C_in, H, W = x.shape
        C_out1 = conv1_weight.shape[1]
        C_out2 = conv2_weight.shape[1]
        H_out1 = H
        W_out1 = W
        H_out2 = H_out1
        W_out2 = W_out1

        # First convolution: out1[B, C_out1, H, W]
        out1 = torch.empty((B, C_out1, H_out1, W_out1), device=x.device, dtype=torch.float32)
        w1_flat = conv1_weight.reshape(C_in, C_out1, 9).contiguous().reshape(-1).to(torch.float32)

        # Grid for conv: (B, C_out, tiles_y, tiles_x)
        tiles_y = (H_out1 + self.BLOCK_H - 1) // self.BLOCK_H
        tiles_x = (W_out1 + self.BLOCK_W - 1) // self.BLOCK_W
        grid1 = (B, C_out1, tiles_y, tiles_x)
        conv3x3_stride1_pad1_tiled_kernel[grid1](
            x, w1_flat, out1,
            B, C_in, C_out1, H, W, H_out1, W_out1,
            BLOCK_H=self.BLOCK_H, BLOCK_W=self.BLOCK_W,
        )

        # GroupNorm 1
        out1_norm = torch.empty_like(out1)
        group_size1 = C_out1 // self.num_groups
        grid_gn1 = (B * self.num_groups,)
        groupnorm_kernel[grid_gn1](
            out1, out1_norm,
            norm1_weight, norm1_bias,
            B, C_out1, H_out1, W_out1,
            group_size1,
            num_groups=self.num_groups,
            eps=self.eps,
            N_INPUT=B * C_out1 * H_out1 * W_out1,
            BLOCK=self.block_size,
        )

        # SiLU 1 (1D)
        out1_silu = torch.empty_like(out1_norm)
        total1 = out1_norm.numel()
        grid_silu1 = ((total1 + self.block_size - 1) // self.block_size,)
        silu_kernel_1d[grid_silu1](out1_norm.reshape(-1), out1_silu.reshape(-1), total1, BLOCK=self.block_size)

        # Second convolution: out2[B, C_out2, H, W]
        out2 = torch.empty((B, C_out2, H_out2, W_out2), device=x.device, dtype=torch.float32)
        w2_flat = conv2_weight.reshape(C_out1, C_out2, 9).contiguous().reshape(-1).to(torch.float32)

        tiles_y2 = (H_out2 + self.BLOCK_H - 1) // self.BLOCK_H
        tiles_x2 = (W_out2 + self.BLOCK_W - 1) // self.BLOCK_W
        grid2 = (B, C_out2, tiles_y2, tiles_x2)
        conv3x3_stride1_pad1_tiled_kernel[grid2](
            out1_silu, w2_flat, out2,
            B, C_out1, C_out2, H_out1, W_out1, H_out2, W_out2,
            BLOCK_H=self.BLOCK_H, BLOCK_W=self.BLOCK_W,
        )

        # GroupNorm 2
        out2_norm = torch.empty_like(out2)
        group_size2 = C_out2 // self.num_groups
        grid_gn2 = (B * self.num_groups,)
        groupnorm_kernel[grid_gn2](
            out2, out2_norm,
            norm2_weight, norm2_bias,
            B, C_out2, H_out2, W_out2,
            group_size2,
            num_groups=self.num_groups,
            eps=self.eps,
            N_INPUT=B * C_out2 * H_out2 * W_out2,
            BLOCK=self.block_size,
        )

        # SiLU 2 (1D)
        out2_silu = torch.empty_like(out2_norm)
        total2 = out2_norm.numel()
        grid_silu2 = ((total2 + self.block_size - 1) // self.block_size,)
        silu_kernel_1d[grid_silu2](out2_norm.reshape(-1), out2_silu.reshape(-1), total2, BLOCK=self.block_size)

        # Add residual x -> out2_silu + x
        # Ensure x is fp32 for the addition kernel
        x_fp32 = x.to(torch.float32).contiguous()
        final = torch.empty_like(out2_silu)
        total_add = out2_silu.numel()
        grid_add = ((total_add + self.block_size - 1) // self.block_size,)
        add_residual_kernel_1d[grid_add](out2_silu.reshape(-1), x_fp32.reshape(-1), total_add, BLOCK=self.block_size)

        return final


def run(*args):
    return ModelNew()(*args)
