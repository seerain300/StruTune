import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: compute per (n, group) sum and sumsq; also compute mean
@triton.jit
def groupnorm_sums_mean_kernel(
    x_ptr, sums_ptr, sumsq_ptr, mean_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    group_size = C_PER_GROUP * H * W
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32 expected
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)
    mean = s / group_size
    tl.store(mean_ptr + out_idx, mean)


# Triton kernel: compute invstd = 1/sqrt(var + eps) per (n, group)
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    mean = tl.load(mean_ptr + out_idx)
    group_size = C_PER_GROUP * H * W
    var = s2 / group_size - mean * mean
    eps = 1e-5
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply GroupNorm (using mean and invstd) + affine + SiLU
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    mean = tl.load(mean_ptr + out_idx)
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        scale = tl.load(norm_w_ptr + ci)
        bias = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                y = (x_val - mean) * invstd  # normalized per (n, group)
                # affine
                y = y * scale + bias
                # SiLU activation: y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                out_idx_final = ((n * C + ci) * H + h) * W + w
                tl.store(out_ptr + out_idx_final, out_val)


def _groupnorm_triton(x, norm_weight, norm_bias, num_groups: int, eps: float):
    """
    Triton implementation of GroupNorm(num_groups, affine) + SiLU.
    x: (B, C, H, W) float32 contiguous
    norm_weight, norm_bias: (C,) float32
    returns: (B, C, H, W) float32
    """
    assert x.is_contiguous(), "Input must be contiguous NCHW"
    assert x.dtype == torch.float32, "Use float32 tensors for Triton kernels"
    B, C, H, W = x.shape
    _assert_divisible(C, num_groups)
    C_PER_GROUP = C // num_groups

    # Allocate reduction outputs
    sums = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    sumsq = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    mean = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    invstd = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)

    # Kernel 1: compute sums and mean
    grid = (B * num_groups,)
    groupnorm_sums_mean_kernel[grid](
        x, sums, sumsq, mean,
        B, C, H, W, num_groups,
        C_PER_GROUP=C_PER_GROUP,
    )

    # Kernel 2: compute invstd
    groupnorm_invstd_kernel[grid](
        sums, sumsq, mean, invstd,
        B, C, H, W, num_groups,
        C_PER_GROUP=C_PER_GROUP,
    )

    # Allocate output
    out = torch.empty_like(x)

    # Kernel 3: normalize + affine + SiLU
    groupnorm_silu_apply_kernel[grid](
        x, norm_weight, norm_bias, out, mean, invstd,
        B, C, H, W, num_groups,
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
    Triton performs GroupNorm + SiLU per stage; convs are done by PyTorch (cuDNN).
    """
    num_groups = 32
    # Ensure inputs are float32 and contiguous
    x = x.contiguous().float()
    # conv1
    out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    # GroupNorm + SiLU stage 1
    out1 = _groupnorm_triton(out1, norm1_weight, norm1_bias, num_groups, eps)
    # conv2
    out2 = torch.nn.functional.conv2d(out1, conv2_weight, bias=None, stride=1, padding=1)
    # GroupNorm + SiLU stage 2
    out2 = _groupnorm_triton(out2, norm2_weight, norm2_bias, num_groups, eps)
    # Residual connection
    out = out2 + x
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
