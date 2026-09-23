import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: 2D convolution with 3x3, stride=1, padding=1, no bias.
# Input: x[B, C_in, H, W], weight[C_in, C_out, 3, 3], output[B, C_out, H, W]
# We pass weights flattened as length = C_in * C_out * 9.
@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,              # *const float
    w_ptr,              # *const float, flattened weights
    out_ptr,            # *float
    B, C_in, C_out, H, W, H_out, W_out,
    N_INPUT,            # int: B*C_in*H*W
    N_WEIGHTS,          # int: C_in*C_out*9
    BLOCK: tl.constexpr,
):
    # Each program computes one output element out[n, co, h_out, w_out]
    pid = tl.program_id(0)
    w_out_idx = pid % W_out
    tmp = pid // W_out
    h_out_idx = tmp % H_out
    tmp = tmp // H_out
    co = tmp % C_out
    n = tmp // C_out

    acc = 0.0

    for ci in range(0, C_in):
        for kh in range(0, 3):
            h_in = h_out_idx + kh - 1  # padding=1
            for kw in range(0, 3):
                w_in = w_out_idx + kw - 1  # padding=1
                # bounds check
                if (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W):
                    x_idx = n * C_in * H * W + ci * H * W + h_in * W + w_in
                    w_idx = (ci * C_out + co) * 9 + (kh * 3 + kw)
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    out_idx = n * C_out * H_out * W_out + co * (H_out * W_out) + h_out_idx * W_out + w_out_idx
    tl.store(out_ptr + out_idx, acc)


# Triton kernel: GroupNorm over NCHW, num_groups is constexpr.
# Assumptions: C_out is divisible by num_groups. Two passes: compute mean/var per (n, group),
# then write normalized outputs using gamma/beta.
@triton.jit
def groupnorm_kernel(
    input_ptr,          # *const float
    output_ptr,         # *float
    gamma_ptr,          # *const float, shape [C_out]
    beta_ptr,           # *const float, shape [C_out]
    B, C_out, H, W,     # int32
    group_size,         # int32 = C_out // num_groups
    num_groups: tl.constexpr,
    eps,                # float32
    N_INPUT,            # int32 = B*C_out*H*W
    BLOCK: tl.constexpr,
):
    # Grid: one program per (n, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    base_c = g * group_size

    # First pass: sum and sumsq per channel in the group
    sum_vec = tl.zeros((group_size,), dtype=tl.float32)
    sumsq_vec = tl.zeros((group_size,), dtype=tl.float32)

    HW = H * W

    for ci in range(0, group_size):
        c = base_c + ci
        sum_c = 0.0
        sumsq_c = 0.0
        for i in range(0, HW):
            h = i // W
            w = i % W
            in_idx = n * C_out * HW + c * HW + h * W + w
            x = tl.load(input_ptr + in_idx)
            sum_c += x
            sumsq_c += x * x
        sum_vec[ci] = sum_c
        sumsq_vec[ci] = sumsq_c

    mean_vec = sum_vec / (HW * group_size)
    var_vec = sumsq_vec / (HW * group_size) - mean_vec * mean_vec
    inv_std_vec = 1.0 / tl.sqrt(var_vec + eps)

    # Second pass: write normalized outputs
    for ci in range(0, group_size):
        c = base_c + ci
        for i in range(0, HW):
            h = i // W
            w = i % W
            in_idx = n * C_out * HW + c * HW + h * W + w
            x = tl.load(input_ptr + in_idx)
            y = (x - mean_vec[ci]) * inv_std_vec[ci]
            gamma = tl.load(gamma_ptr + c)
            beta = tl.load(beta_ptr + c)
            y = y * gamma + beta
            out_idx = n * C_out * HW + c * HW + h * W + w
            tl.store(output_ptr + out_idx, y)


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
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


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
    tl.store(out_ptr + offs, out + res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_size: int = 1024):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_size = block_size

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
        # Input is NCHW
        B, C_in, H, W = x.shape
        # Weight shapes: (C_in, C_out, 3, 3)
        C_out1 = conv1_weight.shape[1]
        C_out2 = conv2_weight.shape[1]
        # Output dims for stride=1, padding=1
        H_out1 = H
        W_out1 = W
        H_out2 = H_out1
        W_out2 = W_out1

        # ------------------------------
        # First convolution: out1[B, C_out1, H, W]
        out1 = torch.empty((B, C_out1, H_out1, W_out1), device=x.device, dtype=x.dtype)

        # Flatten weights: length = C_in * C_out1 * 9
        w1_flat = conv1_weight.reshape(C_in, C_out1, 9).contiguous().reshape(-1)

        grid1 = (B * C_out1 * H_out1 * W_out1,)
        conv3x3_stride1_pad1_kernel[grid1](
            x, w1_flat, out1,
            B, C_in, C_out1, H, W, H_out1, W_out1,
            B * C_in * H * W,
            C_in * C_out1 * 9,
            BLOCK=self.block_size,
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
        out2 = torch.empty((B, C_out2, H_out2, W_out2), device=x.device, dtype=x.dtype)

        # Flatten weights: length = C_out1 * C_out2 * 9
        w2_flat = conv2_weight.reshape(C_out1, C_out2, 9).contiguous().reshape(-1)

        grid2 = (B * C_out2 * H_out2 * W_out2,)
        conv3x3_stride1_pad1_kernel[grid2](
            out1_silu, w2_flat, out2,
            B, C_out1, C_out2, H_out1, W_out1, H_out2, W_out2,
            B * C_out1 * H_out1 * W_out1,
            C_out1 * C_out2 * 9,
            BLOCK=self.block_size,
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
        final = torch.empty_like(out2_silu)
        total_add = out2_silu.numel()
        grid_add = ((total_add + self.block_size - 1) // self.block_size,)
        add_residual_kernel_1d[grid_add](out2_silu.reshape(-1), x.reshape(-1), total_add, BLOCK=self.block_size)

        return final


def run(*args):
    return ModelNew()(*args)
