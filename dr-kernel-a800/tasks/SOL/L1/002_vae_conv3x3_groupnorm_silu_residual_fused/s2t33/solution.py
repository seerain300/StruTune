import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,    # tile size along W
    BLOCK_C: tl.constexpr,    # channels per iteration, specialize to 64
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels in chunks of BLOCK_C (specialize BLOCK_C=64)
    for ci_base in range(0, C, BLOCK_C):
        ci = ci_base
        # Inner vector over channels chunk
        for j in range(0, BLOCK_C):
            # break if ci >= C
            if ci >= C:
                break
            # Accumulate over 3x3 neighborhood (padding=1)
            for dh in range(-1, 1 + 1):
                dh_abs = 1 + dh  # shift by padding
                for dw in range(-1, 1 + 1):
                    dw_abs = 1 + dw  # shift by padding
                    nh = h + dh_abs
                    nw = w_offsets + dw_abs
                    mask_h = (nh >= 0) & (nh < H)
                    mask_hw = mask_w & mask_h
                    # Compute input index for this ci and spatial location
                    in_idx = ((pid_n * C + ci) * H + nh) * W + nw
                    # Load with mask; out-of-bounds -> 0
                    x_val = tl.load(x_ptr + in_idx, mask=mask_hw, other=0.0)
                    # Load weight for (co, ci, dh_abs, dw_abs)
                    w_val = tl.load(w_ptr + (pid_co * C + ci) * 9 + (dh_abs * 3 + dw_abs))
                    acc += x_val * w_val
                    # Note: For 3x3, weight indexing maps (dh_abs, dw_abs) to a linear index 0..8
                    # We pre-accumulate w_val as scalar and multiply vector acc.
            ci += 1

    # Store results
    out_idx = ((pid_n * C_OUT + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


# Triton kernel: compute sum and sum of squares per (n, group)
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    s = 0.0
    s2 = 0.0
    # loop over channels in the group
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        # loop over spatial positions
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    out_idx = n * num_groups + g
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for stability
    tl.store(invstd_ptr + out_idx, invstd)


@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr, mean_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    # one program per (n, group), we broadcast invstd and mean across channels
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    invstd = tl.load(invstd_ptr + out_idx)
    mean = tl.load(mean_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        # affine scale and bias
        gamma = tl.load(norm_w_ptr + ci)  # norm weight
        beta = tl.load(norm_b_bias_ptr + ci)   # norm bias
        # Loop over spatial positions
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                # GroupNorm normalized value
                y = (x_val - mean) * invstd
                # affine
                y = y * gamma + beta
                # SiLU: y * sigmoid(y) = y / (1 + exp(-y))
                sig = 1.0 / (1.0 + tl.exp(-y))
                y = y * sig
                tl.store(out_ptr + idx, y)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps=1e-5):
        super().__init__()
        self.conv1_weight = conv1_weight  # shape (C, C, 3, 3)
        self.conv2_weight = conv2_weight  # shape (C, C, 3, 3)
        self.norm1_weight = norm1_weight  # shape (C,)
        self.norm1_bias = norm1_bias      # shape (C,)
        self.norm2_weight = norm2_weight  # shape (C,)
        self.norm2_bias = norm2_bias      # shape (C,)
        self.eps = eps
        num_groups = 32
        _assert_divisible(self.conv1_weight.shape[1], num_groups)
        _assert_divisible(self.conv2_weight.shape[1], num_groups)
        self.num_groups = num_groups
        self.C = self.conv1_weight.shape[1]  # C_in == C_out == C for each conv in the original code
        self.C_PER_GROUP = self.C // self.num_groups  # must be 2 for C=64, num_groups=32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Stage 1: Conv1 via Triton
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid1 = (B, C, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid1](
            x.contiguous().to(torch.float32), self.conv1_weight.contiguous().to(torch.float32), out1,
            B, C, H, W, C,
            BLOCK_W=64, BLOCK_C=64
        )

        # Stage 1: GroupNorm + SiLU via Triton
        out1_fp32 = out1.contiguous()
        sums1 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        sumsq1 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        grid_sums = (B * self.num_groups,)
        groupnorm_sums_kernel[grid_sums](
            out1_fp32, sums1, sumsq1,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )

        invstd1 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums](
            sums1, sumsq1, invstd1,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )

        group_size1 = self.C_PER_GROUP * H * W
        mean1 = (sums1 / group_size1).to(torch.float32)

        out1_norm = torch.empty_like(out1_fp32, device=device, dtype=torch.float32)
        groupnorm_silu_apply_kernel[grid_sums](
            out1_fp32, self.norm1_weight.to(torch.float32), self.norm1_bias.to(torch.float32),
            out1_norm, invstd1, mean1,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )

        # Stage 2: Conv2 via Triton
        out2 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid2 = (B, C, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid2](
            out1_norm.contiguous(), self.conv2_weight.contiguous().to(torch.float32), out2,
            B, C, H, W, C,
            BLOCK_W=64, BLOCK_C=64
        )

        # Stage 2: GroupNorm + SiLU via Triton (same as stage 1)
        out2_fp32 = out2.contiguous()
        sums2 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        sumsq2 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums](
            out2_fp32, sums2, sumsq2,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )
        invstd2 = torch.empty(B * self.num_groups, device=device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums](
            sums2, sumsq2, invstd2,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )
        mean2 = (sums2 / group_size1).to(torch.float32)

        out2_norm = torch.empty_like(out2_fp32, device=device, dtype=torch.float32)
        groupnorm_silu_apply_kernel[grid_sums](
            out2_fp32, self.norm2_weight.to(torch.float32), self.norm2_bias.to(torch.float32),
            out2_norm, invstd2, mean2,
            B, C, H, W, self.num_groups,
            self.C_PER_GROUP
        )

        # Residual add (PyTorch elementwise)
        residual = x.contiguous().to(torch.float32)
        out = out2_norm + residual

        return out


def run(*args):
    return ModelNew()(*args)
