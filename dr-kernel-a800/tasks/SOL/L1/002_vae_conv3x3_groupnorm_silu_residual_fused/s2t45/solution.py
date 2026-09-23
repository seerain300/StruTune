import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: compute sum and sum of squares per (n, group) across channels in the group and all H*W elements
# Assumes C is divisible by num_groups (32 here), and H, W are static compile-time loops.
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,  # == C // num_groups
    H_CONST: tl.constexpr,      # compile-time H
    W_CONST: tl.constexpr,      # compile-time W
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    # Loop over channels in this group
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        # Loop over all spatial positions (H, W) — static, compile-time
        for h in range(0, H_CONST):
            for w in range(0, W_CONST):
                idx = ((n * C + ci) * H_CONST + h) * W_CONST + w
                x_val = tl.load(x_ptr + idx)  # float32 expected
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute invstd = 1/sqrt(var + eps) per (n, group) using sums and sumsq
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # numerical stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using precomputed invstd
# SiLU(x) = x * sigmoid(x)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    # Recompute mean per (n, group) to normalize. This kernel will read x again; H/W are static.
    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
    group_size = C_PER_GROUP * H * W
    mean = s / group_size

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        scale = tl.load(norm_w_ptr + ci)  # (C,)
        bias = tl.load(norm_b_ptr + ci)   # (C,)
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                # GroupNorm: (x - mean) * invstd
                normed = (x_val - mean) * invstd
                # Apply affine
                y = normed * scale + bias
                # SiLU
                sig = 1.0 / (1.0 + tl.exp(-y))
                silu = y * sig
                tl.store(out_ptr + idx, silu)


def _run_groupnorm_silu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int = 32, eps: float = 1e-5) -> torch.Tensor:
    # x: (B, C, H, W) float32
    B, C, H, W = x.shape
    _assert_divisible(C, num_groups)
    C_PER_GROUP = C // num_groups

    # Ensure contiguous and float32
    x_in = x.contiguous()
    if x_in.dtype != torch.float32:
        x_in = x_in.float()

    # Allocate outputs for sums, sumsq, invstd
    sums = torch.empty(B * num_groups, device=x_in.device, dtype=torch.float32)
    sumsq = torch.empty(B * num_groups, device=x_in.device, dtype=torch.float32)
    invstd = torch.empty(B * num_groups, device=x_in.device, dtype=torch.float32)

    # Pass H, W as constexpr for Triton (compile-time loops)
    H_CONST = H
    W_CONST = W

    # Kernel 1: compute sums and sumsq
    grid_sums = (B * num_groups,)
    groupnorm_sums_kernel[grid_sums](
        x_in, sums, sumsq,
        B, C, H_CONST, W_CONST, num_groups,
        C_PER_GROUP=C_PER_GROUP,
        H_CONST=H_CONST, W_CONST=W_CONST,
    )

    # Kernel 2: compute invstd
    groupnorm_invstd_kernel[grid_sums](
        sums, sumsq, invstd,
        B, C, H_CONST, W_CONST, num_groups,
        C_PER_GROUP=C_PER_GROUP,
    )

    # Allocate output tensor
    out = torch.empty_like(x_in)

    # Kernel 3: apply normalization + affine + SiLU
    groupnorm_silu_apply_kernel[grid_sums](
        x_in, weight, bias, out, invstd,
        B, C, H_CONST, W_CONST, num_groups,
        C_PER_GROUP=C_PER_GROUP,
    )

    return out


@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Triton is used for GroupNorm + SiLU per stage. Conv3x3 is done via PyTorch (cuDNN) for correctness and speed.
    """
    # First conv (stride=1, padding=1)
    out = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

    # GroupNorm + SiLU stage 1 using Triton
    out = _run_groupnorm_silu_triton(out, norm1_weight, norm1_bias, num_groups=32, eps=eps)

    # Second conv (stride=1, padding=1)
    out = torch.nn.functional.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)

    # GroupNorm + SiLU stage 2 using Triton
    out = _run_groupnorm_silu_triton(out, norm2_weight, norm2_bias, num_groups=32, eps=eps)

    # Residual connection
    out = out + x

    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
