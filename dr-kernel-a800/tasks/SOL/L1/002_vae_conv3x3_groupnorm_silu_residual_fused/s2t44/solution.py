import torch
import triton
import triton.language as tl


# Triton kernel: compute sum and sum of squares per (n, group) across channels and spatial elements
# x is NCHW contiguous, we compute over all channels in the group and all H*W elements for that sample.
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    start_ci = g * C_PER_GROUP
    group_size_elems = C_PER_GROUP * H * W
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(0, C_PER_GROUP):
        ci_abs = start_ci + ci
        # loop over H and W (compile-time)
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci_abs) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    out_idx = n * num_groups + g
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, group) from sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size_elems = C_PER_GROUP * H * W
    mean = s / group_size_elems
    var = s2 / group_size_elems - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply GroupNorm (using precomputed mean and invstd) + affine + SiLU
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, y_ptr, norm_w_ptr, norm_b_ptr, invstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(0, C_PER_GROUP):
        ci_abs = start_ci + ci
        w = tl.load(norm_w_ptr + ci_abs)
        b = tl.load(norm_b_ptr + ci_abs)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx_in = ((n * C + ci_abs) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx_in)
                # normalize
                norm = (x_val - (s - b) * invstd)  # mean is s/group_size; affine is b
                # SiLU: x * sigmoid(x)
                sig = 1.0 / (1.0 + tl.exp(-norm))
                y_val = norm * sig
                # apply affine weight w
                y_val = y_val * w + b
                idx_out = ((n * C + ci_abs) * H + h) * W + w_idx
                tl.store(y_ptr + idx_out, y_val)


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure dtype and layout
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, C, H, W = x.shape
        _assert_divisible(C, 32)
        C_OUT1 = conv1_weight.shape[0]  # usually equals C (64)
        C_OUT2 = conv2_weight.shape[0]  # usually equals C (64)

        # First conv
        # F.conv2d handles stride=1, padding=1, no bias; we leave it in PyTorch for robustness
        out = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm + SiLU stage 1 in Triton
        num_groups = 32
        C_PER_GROUP = C // num_groups

        # Allocate sums and invstd buffers (float32 for stability)
        sums = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)
        invstd = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)

        # Run reduction kernel
        grid_sums = (B * num_groups,)
        groupnorm_sums_kernel[grid_sums](
            out, sums, sumsq,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Compute invstd
        groupnorm_invstd_kernel[grid_sums](
            sums, sumsq, invstd,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Apply kernel
        out1a = torch.empty_like(out)
        groupnorm_silu_apply_kernel[grid_sums](
            out, out1a, norm1_weight, norm1_bias, invstd,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Second conv
        out = torch.nn.functional.conv2d(out1a, conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm + SiLU stage 2 in Triton
        sums2 = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)
        sumsq2 = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)
        invstd2 = torch.empty(B * num_groups, dtype=torch.float32, device=x.device)

        # Run reduction kernel (second stage)
        groupnorm_sums_kernel[grid_sums](
            out, sums2, sumsq2,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        groupnorm_invstd_kernel[grid_sums](
            sums2, sumsq2, invstd2,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Apply kernel (second stage)
        out2a = torch.empty_like(out)
        groupnorm_silu_apply_kernel[grid_sums](
            out, out2a, norm2_weight, norm2_bias, invstd2,
            B=B, C=C, H=H, W=W,
            num_groups=num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Residual connection
        out = out2a + x
        return out


def run(*args):
    return ModelNew()(*args)
