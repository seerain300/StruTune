import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: compute per-(n, group) sum and sum of squares for GroupNorm
# Grid: (B * num_groups,)
@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    s = 0.0
    s2 = 0.0
    # loop over channels in the group and spatial positions
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, group) from sums and sumsq
# Grid: (B * num_groups,)
@triton.jit
def groupnorm_invstd(
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


# Triton kernel: apply GroupNorm (using mean and invstd) + affine + SiLU
# We need a grid with 3 dims. We will use (B, C, H*W) and derive (n, group) from pid_ci via group = ci // C_PER_GROUP.
# However, Triton doesn't allow indexing mean/invstd by pid_ci directly. Therefore, we use grid (B, num_groups, H*W) and derive n and group from pid_n, pid_group respectively.
@triton.jit
def groupnorm_silu_apply_group(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_group = tl.program_id(1)  # group index
    pid_hw = tl.program_id(2)     # linear index over H*W

    h = pid_hw // W
    w = pid_hw % W

    # For each channel in this group, we process (h, w). We loop over channels.
    for ci in range(pid_group * C_PER_GROUP, (pid_group + 1) * C_PER_GROUP):
        # mean/invstd are per (n, group)
        mean = tl.load(mean_ptr + (pid_n * num_groups + pid_group))
        invstd = tl.load(invstd_ptr + (pid_n * num_groups + pid_group))

        scale = tl.load(norm_w_ptr + ci)  # per-channel affine scale
        bias = tl.load(norm_b_ptr + ci)   # per-channel affine bias

        # Load input x[n, ci, h, w]
        idx = ((pid_n * C + ci) * H + h) * W + w
        x_val = tl.load(x_ptr + idx)

        # Normalize
        y = (x_val - mean) * invstd
        # Affine
        y = y * scale + bias
        # SiLU: y * sigmoid(y)
        sig = 1.0 / (1.0 + tl.exp(-y))
        out_val = y * sig

        # Store output
        out_idx = ((pid_n * C + ci) * H + h) * W + w
        tl.store(out_ptr + out_idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Args:
            x: Input tensor of shape (B, C, H, W)
            conv1_weight: First conv weights (C, C, 3, 3)
            norm1_weight: First GroupNorm scale (C,)
            norm1_bias: First GroupNorm bias (C,)
            conv2_weight: Second conv weights (C, C, 3, 3)
            norm2_weight: Second GroupNorm scale (C,)
            norm2_bias: Second GroupNorm bias (C,)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Validate shapes
        B, C, H, W = x.shape
        _assert_divisible(C, self.num_groups)
        C_PER_GROUP = C // self.num_groups

        # First stage: Conv3x3 -> GroupNorm -> SiLU
        # Use PyTorch conv2d (cuDNN) for correctness and speed
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Ensure float32 for Triton
        if out.dtype != torch.float32:
            out = out.float()

        # Allocate reduction buffers and launch Triton kernels for GroupNorm + SiLU
        BxG = B * self.num_groups
        sums = torch.empty(BxG, device=out.device, dtype=torch.float32)
        sumsq = torch.empty(BxG, device=out.device, dtype=torch.float32)
        invstd = torch.empty(BxG, device=out.device, dtype=torch.float32)

        # 1) Reduce sums and sumsq
        groupnorm_reduce_sums[(BxG,)](out, sums, sumsq, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)

        # 2) Compute invstd
        groupnorm_invstd[(BxG,)](sums, sumsq, invstd, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)

        # 3) Apply GroupNorm + affine + SiLU
        # We need output tensor for the normalized+activated result
        out_norm = torch.empty_like(out)
        # Launch apply kernel: grid (B, num_groups, H*W)
        grid = (B, self.num_groups, H * W)
        groupnorm_silu_apply_group[grid](out, norm1_weight, norm1_bias, out_norm, sums, invstd, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)

        # Second stage: Conv3x3 -> GroupNorm -> SiLU
        out2 = F.conv2d(out_norm, conv2_weight, bias=None, stride=1, padding=1)
        if out2.dtype != torch.float32:
            out2 = out2.float()

        # GroupNorm + SiLU for second stage
        sums2 = torch.empty(BxG, device=out2.device, dtype=torch.float32)
        sumsq2 = torch.empty(BxG, device=out2.device, dtype=torch.float32)
        invstd2 = torch.empty(BxG, device=out2.device, dtype=torch.float32)

        groupnorm_reduce_sums[(BxG,)](out2, sums2, sumsq2, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)
        groupnorm_invstd[(BxG,)](sums2, sumsq2, invstd2, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)

        out_norm2 = torch.empty_like(out2)
        grid2 = (B, self.num_groups, H * W)
        groupnorm_silu_apply_group[grid2](out2, norm2_weight, norm2_bias, out_norm2, sums2, invstd2, B, C, H, W, self.num_groups, C_PER_GROUP=C_PER_GROUP)

        # Residual add: out = out_norm2 + x
        # x may be different device; ensure it's on the same device and dtype
        if x.device != out_norm2.device:
            x = x.to(out_norm2.device)
        if x.dtype != out_norm2.dtype:
            x = x.float()
        out_final = out_norm2 + x

        return out_final


def run(*args):
    return ModelNew()(*args)
