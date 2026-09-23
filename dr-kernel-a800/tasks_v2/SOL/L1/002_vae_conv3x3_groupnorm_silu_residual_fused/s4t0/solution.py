import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: GroupNorm (per sample, per group, per channel)
# We perform two passes:
#  - Pass 1: compute per-channel mean and rstd across the group's spatial and channel extents
#  - Pass 2: normalize and apply per-channel affine (scale, bias)
# Assumptions:
#  - Input tensor y is (B, C, H, W), contiguous NCHW
#  - gamma, beta are (C,), contiguous
#  - num_groups is fixed (here 32), C % num_groups == 0
@triton.jit
def group_norm_triton_kernel(
    y_ptr,               # *float32, input
    out_ptr,             # *float32, output
    gamma_ptr,           # *float32, per-channel scale
    beta_ptr,            # *float32, per-channel bias
    B: tl.constexpr,     # int
    C: tl.constexpr,     # int
    H: tl.constexpr,     # int
    W: tl.constexpr,     # int
    num_groups: tl.constexpr,  # int, here 32
    eps: tl.constexpr,          # float
):
    # Each program instance handles one (n, group)
    n = tl.program_id(0)
    group_id = tl.program_id(1)
    if n >= B:
        return

    # Compute group size in channels (constant per group)
    channels_per_group = C // num_groups

    # First pass: compute sum and sum of squares per channel in the group
    sum_vec = tl.zeros((channels_per_group,), dtype=tl.float32)
    sumsq_vec = tl.zeros((channels_per_group,), dtype=tl.float32)

    # Loop over channels within the group
    for ch_in_group in range(channels_per_group):
        c = group_id * channels_per_group + ch_in_group

        # Accumulate sum and sum of squares across H*W for this channel c
        hw_count = H * W
        total = 0.0
        total_sq = 0.0
        # For each spatial position
        for hw in range(hw_count):
            h = hw // W
            w = hw % W
            idx = ((n * C + c) * (H * W)) + hw
            val = tl.load(y_ptr + idx)
            total += val
            total_sq += val * val

        sum_vec[ch_in_group] = total
        sumsq_vec[ch_in_group] = total_sq

    # Compute mean and rstd per channel in the group
    hw_per_group = H * W
    count_per_channel = hw_per_group * channels_per_group  # total elements per (n, group, c)
    mean_vec = sum_vec / count_per_channel
    var_vec = sumsq_vec / count_per_channel - mean_vec * mean_vec
    rstd_vec = 1.0 / tl.sqrt(var_vec + eps)

    # Second pass: normalize and apply affine
    for ch_in_group in range(channels_per_group):
        c = group_id * channels_per_group + ch_in_group
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        for hw in range(hw_count):
            h = hw // W
            w = hw % W
            idx = ((n * C + c) * (H * W)) + hw
            val = tl.load(y_ptr + idx)
            normalized = (val - mean_vec[ch_in_group]) * rstd_vec[ch_in_group]
            out_val = normalized * gamma + beta
            tl.store(out_ptr + idx, out_val)


# Triton kernel: SiLU activation (y = x * sigmoid(x))
@triton.jit
def silu_triton_kernel(x_ptr, out_ptr, N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    # One program instance per element
    idx = tl.program_id(0)
    if idx >= N * C * H * W:
        return
    x = tl.load(x_ptr + idx)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + idx, y)


def group_norm_triton(y: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float) -> torch.Tensor:
    """
    Triton implementation of GroupNorm over (B, C, H, W) with given gamma/beta and num_groups.
    Assumes:
      - y is float32, CUDA, contiguous NCHW
      - gamma, beta are float32, shape (C,)
      - num_groups is fixed (here 32) and C % num_groups == 0
    Returns:
      - out tensor same shape as y
    """
    assert TRITON_AVAILABLE, "Triton not available"
    assert y.is_cuda, "Input tensor must be on CUDA for Triton"
    assert y.dtype == torch.float32, "Expected float32 tensor"
    assert y.is_contiguous(), "Input must be contiguous"
    B, C, H, W = y.shape
    assert C % num_groups == 0, f"num_groups={num_groups} must divide C={C}"
    assert gamma.shape == (C,), f"gamma shape must be (C,), got {gamma.shape}"
    assert beta.shape == (C,), f"beta shape must be (C,), got {beta.shape}"

    out = torch.empty_like(y)

    # Launch grid: one program per (n, group)
    grid = (B, num_groups)
    group_norm_triton_kernel[grid](
        y, out, gamma, beta,
        B, C, H, W, num_groups, eps,
        num_warps=4,  # heuristic
        num_stages=2  # heuristic
    )
    return out


def silu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of SiLU activation.
    """
    assert TRITON_AVAILABLE, "Triton not available"
    assert x.is_cuda, "Input tensor must be on CUDA for Triton"
    assert x.dtype == torch.float32, "Expected float32 tensor"
    assert x.is_contiguous(), "Input must be contiguous"
    N = x.numel()
    out = torch.empty_like(x)
    grid = (N,)
    silu_triton_kernel[grid](
        x, out, N, x.shape[1], x.shape[2], x.shape[3],
        num_warps=4,
        num_stages=2
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is provided at runtime in forward (as in original)

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Triton replaces GroupNorm and SiLU. Convolutions are left to PyTorch (cuDNN).
        """
        # Ensure inputs are contiguous (NCHW)
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # First conv
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm 1 (Triton), then SiLU (Triton)
        out = group_norm_triton(out, norm1_weight, norm1_bias, num_groups=32, eps=eps)
        out = silu_triton(out)

        # Second conv
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm 2 (Triton), then SiLU (Triton)
        out = group_norm_triton(out, norm2_weight, norm2_bias, num_groups=32, eps=eps)
        out = silu_triton(out)

        # Residual add
        out = out + x

        return out


def run(*args):
    return ModelNew()(*args)
