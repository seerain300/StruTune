import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernel for GroupNorm over NCHW, with num_groups as a constexpr.
# Assumptions:
# - Input/output tensors are contiguous NCHW.
# - C is divisible by num_groups.
# - The kernel receives the per-channel scale (gamma) and bias (beta) of size C.
# - eps is scalar float.
# - We perform two passes: first to compute mean/var per (n, channel_in_group), second to write normalized outputs.
@triton.jit
def groupnorm_kernel(
    input_ptr,          # *const float
    output_ptr,         # *float
    gamma_ptr,          # *const float, shape [C]
    beta_ptr,           # *const float, shape [C]
    N, C, H, W,         # int32
    group_size,         # int32 = C // num_groups
    num_groups: tl.constexpr,       # compile-time constant num_groups
    eps: tl.constexpr,              # compile-time constant eps (for type consistency)
):
    # Each program handles one (n, channel_in_group) pair
    # grid = (N * num_groups,)
    pid = tl.program_id(0)
    n = pid // num_groups
    group_id = pid % num_groups

    # Total number of elements in this group across all channels and spatial
    # group_size is fixed per group, and H*W is the spatial extent; we need to iterate across spatial and channels in the group.
    HW = H * W
    total_elems = group_size * HW

    # Accumulate sum and sum of squares for this (n, group_id)
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over channels in the group
    # Note: Triton loops over static ranges are best; we can use a while-like loop by iterating channel index.
    c_start = group_id * group_size
    # For each channel in group, loop over spatial HW
    for ci in range(group_size):
        c = c_start + ci
        # Accumulate over spatial
        # Compute base offset for (n, c, :, :)
        # offset = ((n * C + c) * H + h) * W + w
        # We will compute via strides: since tensor is contiguous, we can compute linear index via (n*C + c) * (H*W)
        base_n = n * C
        base_nc = base_n + c
        base_offset = base_nc * (H * W)

        # Since we can't directly loop h,w in Triton without an index vector, we iterate in chunks.
        # We'll accumulate manually for simplicity. Triton supports scalar loops; we will do elementwise addressing by offset.
        # Alternative approach: use tl.arange to vectorize and sum in chunks, but Triton does not support dynamic indexing into 1D ranges across arbitrary strides. So we do per-pixel accumulation.
        # For practicality and correctness, we do it per-pixel with while-like iteration. Triton supports for-range over runtime values using tl.range-like constructs; here we use Python range with runtime bounds.
        # We'll restructure by iterating over HW in chunks of 1 (since Triton scalar loop is fine), but better to do vectorized processing: build a [HW] vector of linear indices and reduce.
        # However, Triton reduction across a vector requires vector ops. Simpler: use two passes explicitly:
        # Pass 1: compute sum and sum of squares; Pass 2: write normalized values.

        # Implement pass 1: compute sum and sum of squares
        for i in range(HW):
            # offset = base_offset + i
            offset = base_offset + i
            val = tl.load(input_ptr + offset)
            sum_val += val
            sum_sq += val * val

    # Compute mean and variance for this (n, group_id)
    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: write normalized output with affine gamma/beta
    for ci in range(group_size):
        c = c_start + ci
        base_nc = (n * C) + c
        base_offset = base_nc * (H * W)

        for i in range(HW):
            offset = base_offset + i
            x = tl.load(input_ptr + offset)
            # Normalize: (x - mean) * inv_std
            y = (x - mean) * inv_std
            gamma = tl.load(gamma_ptr + c)
            beta = tl.load(beta_ptr + c)
            out = y * gamma + beta
            tl.store(output_ptr + offset, out)


# Triton kernel for SiLU: y = x * sigmoid(x), elementwise.
@triton.jit
def silu_kernel(input_ptr, output_ptr, N, C, H, W):
    HW = H * W
    # Grid can be (N, C, H, W) or we use a 1D mapping; since we want simple elementwise, use a 1D grid over N*C*H*W
    grid_elems = N * C * H * W
    pid = tl.program_id(0)
    # Each program handles one element
    if pid < grid_elems:
        # Compute n, c, h, w from pid
        # Note: Triton doesn't support Pythonic integer division directly here; we use integer operations:
        tmp = pid // (C * H * W)
        rem = pid % (C * H * W)
        n = tmp
        chw = C * H * W
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W
        offset = (n * C + c) * (H * W) + h * W + w
        x = tl.load(input_ptr + offset)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(output_ptr + offset, y)


# Triton kernel for elementwise addition: out = out + residual
@triton.jit
def add_residual_kernel(out_ptr, residual_ptr, N, C, H, W):
    HW = H * W
    grid_elems = N * C * H * W
    pid = tl.program_id(0)
    if pid < grid_elems:
        tmp = pid // (C * H * W)
        rem = pid % (C * H * W)
        n = tmp
        chw = C * H * W
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W
        offset = (n * C + c) * (H * W) + h * W + w
        a = tl.load(out_ptr + offset)
        b = tl.load(residual_ptr + offset)
        tl.store(out_ptr + offset, a + b)


class ModelNew(nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

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
        # Ensure tensors are on CUDA and contiguous
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        x = x.contiguous()

        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm."

        # 1) First conv
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out1 = out1.contiguous()

        # 2) GroupNorm on out1, then SiLU
        out1_norm = torch.empty_like(out1)
        group_size = C // self.num_groups
        # Launch Triton kernel: grid over N * num_groups
        grid = (B * self.num_groups,)
        groupnorm_kernel[grid](
            out1, out1_norm,
            norm1_weight, norm1_bias,
            B, C, H, W,
            group_size,
            num_groups=self.num_groups,
            eps=self.eps,  # constexpr eps
        )

        out1_silu = torch.empty_like(out1_norm)
        silu_kernel[B * C * H * W](out1_norm, out1_silu, B, C, H, W)

        # 3) Second conv
        out2 = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)
        out2 = out2.contiguous()

        # 4) GroupNorm on out2, then SiLU
        out2_norm = torch.empty_like(out2)
        grid2 = (B * self.num_groups,)
        groupnorm_kernel[grid2](
            out2, out2_norm,
            norm2_weight, norm2_bias,
            B, C, H, W,
            group_size,
            num_groups=self.num_groups,
            eps=self.eps,
        )

        out2_silu = torch.empty_like(out2_norm)
        silu_kernel[B * C * H * W](out2_norm, out2_silu, B, C, H, W)

        # 5) Add residual x
        final = torch.empty_like(out2_silu)
        add_residual_kernel[B * C * H * W](out2_silu, x, B, C, H, W)

        return final


def run(*args):
    return ModelNew()(*args)
