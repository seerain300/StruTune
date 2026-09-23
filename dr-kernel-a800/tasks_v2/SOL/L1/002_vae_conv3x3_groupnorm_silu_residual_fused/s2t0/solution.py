import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Kernel 1: Conv2d 3x3, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    KH: tl.constexpr, KW: tl.constexpr,  # 3x3
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,  # 1
    BLOCK_W: tl.constexpr  # e.g., 64
):
    # program ids: over batch and output channels; we vectorize over W
    b = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    grid_w = tl.num_programs(2)  # corresponds to number of tiles along W
    wo_block = tl.program_id(3)  # which W tile this program computes
    # vector of W indices for this program
    offs_w = wo_block * BLOCK_W + tl.arange(0, BLOCK_W)
    # mask to handle boundaries
    mask_w = offs_w < W

    # Accumulator for BLOCK_W outputs
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over input channels and 3x3 neighborhood
    # For each ci, compute contribution to all BLOCK_W outputs (fixed ho, vary wo)
    for ci in range(0, C):
        # For each kernel position
        for kh in range(0, KH):
            hi = ho + kh - PAD_H
            # valid_h: scalar mask
            valid_h = (hi >= 0) & (hi < H)
            for kw in range(0, KW):
                wi = offs_w + kw - PAD_W  # vector
                valid_w = (wi >= 0) & (wi < W)
                # combined mask
                mask = mask_w & valid_w & valid_h
                # load x[b, ci, hi, wi]
                # NCHW strides: x strides are (C*H*W, H*W, W, 1)
                x_index = ((b * C) + ci) * (H * W) + hi * W + wi
                x_val = tl.load(x_ptr + x_index, mask=mask, other=0.0)
                # load weight[co, ci, kh, kw] (scalar)
                w_index = co * (C * KH * KW) + ci * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_index)  # scalar
                acc += x_val * w_val  # vectorized over offs_w

    # Write results to out[b, co, ho, offs_w]
    out_index = ((b * C_OUT) + co) * (H * W) + ho * W + offs_w
    tl.store(out_ptr + out_index, acc, mask=mask_w)


# GroupNorm reduction kernels
# 1) Compute sum and sumsq per (n, group)
@triton.jit
def group_sums_kernel(x_ptr, sums_ptr, sumsq_ptr, B, C, HW, num_groups, eps):
    # grid: (B * num_groups,)
    pid = tl.program_id(0)
    b = pid // num_groups
    g = pid % num_groups
    C_per_group = C // num_groups
    ci_start = g * C_per_group
    ci_end = (g + 1) * C_per_group

    # Accumulators
    total_sum = 0.0
    total_sumsq = 0.0

    # Loop over channels in this group
    for ci in range(ci_start, ci_end):
        base = ((b * C) + ci) * HW
        # loop over H*W
        for k in range(0, HW):
            x_val = tl.load(x_ptr + base + k)
            total_sum += x_val
            total_sumsq += x_val * x_val

    # Store sums
    index = b * num_groups + g
    tl.store(sums_ptr + index, total_sum)
    tl.store(sumsq_ptr + index, total_sumsq)


# 2) Compute inv std per (n, group) from sums and sumsq
@triton.jit
def group_invstd_kernel(sums_ptr, sumsq_ptr, invstd_ptr, B, C, HW, num_groups, eps):
    pid = tl.program_id(0)
    b = pid // num_groups
    g = pid % num_groups
    index = b * num_groups + g
    sum_val = tl.load(sums_ptr + index)
    sumsq_val = tl.load(sumsq_ptr + index)
    n = (C // num_groups) * HW
    mean = sum_val / n
    var = sumsq_val / n - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + index, invstd)


# 3) Normalize + affine + SiLU per (n, group). SiLU: y = x * sigmoid(x)
@triton.jit
def groupnorm_silu_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, invstd_ptr, B, C, HW, num_groups):
    pid = tl.program_id(0)
    b = pid // num_groups
    g = pid % num_groups
    C_per_group = C // num_groups
    ci_start = g * C_per_group
    ci_end = (g + 1) * C_per_group
    invstd = tl.load(invstd_ptr + (b * num_groups + g))

    # For each channel in this group, loop over H*W elements
    for ci in range(ci_start, ci_end):
        base_x = ((b * C) + ci) * HW
        base_out = ((b * C) + ci) * HW
        # load scale and bias
        gamma = tl.load(weight_ptr + ci)
        beta = tl.load(bias_ptr + ci)
        for k in range(0, HW):
            x_val = tl.load(x_ptr + base_x + k)
            y = ((x_val - mean) * invstd) * gamma + beta
            # SiLU
            s = 1.0 / (1.0 + tl.exp(-y))
            z = y * s
            tl.store(out_ptr + base_out + k, z)


# ModelNew: Triton-optimized version
class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Validate shapes
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "num_groups must divide C"
        C_OUT = C  # output channels equal to input channels

        # Ensure CUDA and contiguity; use float32 for compute
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors")
        x = x.contiguous().to(torch.float32)

        # Allocate output tensors
        out1 = torch.empty_like(x, dtype=torch.float32, device=x.device)
        out2 = torch.empty_like(x, dtype=torch.float32, device=x.device)

        # Launch conv1 kernel
        grid_conv = (B, C_OUT, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid_conv](
            x, conv1_weight, out1,
            B, C, H, W, C_OUT,
            KH=3, KW=3, PAD_H=1, PAD_W=1,
            BLOCK_W=64
        )

        # GroupNorm + SiLU for out1
        HW = H * W
        C_per_group = C // self.num_groups
        n_elements_groups = self.num_groups  # B * num_groups programs for sums
        sums = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        group_sums_kernel[(B * self.num_groups,)](out1, sums, sumsq, B, C, HW, self.num_groups, self.eps)

        invstd = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        group_invstd_kernel[(B * self.num_groups,)](sums, sumsq, invstd, B, C, HW, self.num_groups, self.eps)

        # Now normalize + affine + SiLU
        out1_norm = torch.empty_like(out1, dtype=torch.float32, device=x.device)
        groupnorm_silu_kernel[(B * self.num_groups,)](
            out1, norm1_weight, norm1_bias, out1_norm, invstd,
            B, C, HW, self.num_groups
        )

        # Launch conv2 kernel
        grid_conv2 = (B, C_OUT, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid_conv2](
            out1_norm, conv2_weight, out2,
            B, C, H, W, C_OUT,
            KH=3, KW=3, PAD_H=1, PAD_W=1,
            BLOCK_W=64
        )

        # GroupNorm + SiLU for out2
        sums2 = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        sumsq2 = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        group_sums_kernel[(B * self.num_groups,)](out2, sums2, sumsq2, B, C, HW, self.num_groups, self.eps)

        invstd2 = torch.empty(n_elements_groups, dtype=torch.float32, device=x.device)
        group_invstd_kernel[(B * self.num_groups,)](sums2, sumsq2, invstd2, B, C, HW, self.num_groups, self.eps)

        out2_norm = torch.empty_like(out2, dtype=torch.float32, device=x.device)
        groupnorm_silu_kernel[(B * self.num_groups,)](
            out2, norm2_weight, norm2_bias, out2_norm, invstd2,
            B, C, HW, self.num_groups
        )

        # Add residual x
        out = out2_norm + x

        return out


def run(*args):
    return ModelNew()(*args)
