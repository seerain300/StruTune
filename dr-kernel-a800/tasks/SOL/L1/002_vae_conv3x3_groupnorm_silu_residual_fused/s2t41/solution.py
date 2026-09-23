import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: compute sum and sum of squares per (n, group) across channels and spatial elements.
# Assumes NCHW layout. num_groups, C_PER_GROUP, H, W are constexpr to allow loops.
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    for ci in range(0, C_PER_GROUP):
        ci_abs = start_ci + ci
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci_abs) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, group) from sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for numerical stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, invstd_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(0, C_PER_GROUP):  # channels in this group
        ci_abs = start_ci + ci
        gamma = tl.load(norm_w_ptr + ci_abs)  # norm_weight[ci]
        beta = tl.load(norm_b_ptr + ci_abs)   # norm_bias[ci]
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci_abs) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                y = (x_val - mean) * invstd * gamma + beta  # normalize + affine
                # SiLU: y * sigmoid(y) = y / (1 + exp(-y))
                sig = 1.0 / (1.0 + tl.exp(-y))
                z = y * sig
                tl.store(out_ptr + idx, z)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Validate shapes
        B, C, H, W = x.shape
        _assert_divisible(C, self.num_groups)
        C_PER_GROUP = C // self.num_groups

        # Ensure contiguous and float32 for Triton
        x = x.contiguous().to(torch.float32)
        # Conv1
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + SiLU for out1
        out1_ = torch.empty_like(out1, dtype=torch.float32)
        Bn = out1.shape[0]
        Cn = out1.shape[1]
        Hn = out1.shape[2]
        Wn = out1.shape[3]

        # Buffers for sums and invstd
        sums = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out1.device)
        sumsq = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out1.device)
        invstd = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out1.device)

        # Launch sum kernel
        grid_sums = (Bn * self.num_groups,)
        groupnorm_sums_kernel[grid_sums](
            out1, sums, sumsq,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )
        # Launch invstd kernel
        groupnorm_invstd_kernel[grid_sums](
            sums, sumsq, invstd,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )
        # Launch apply kernel
        groupnorm_silu_apply_kernel[grid_sums](
            out1, out1_, norm1_weight, norm1_bias, invstd,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )

        # Conv2
        out2 = torch.nn.functional.conv2d(out1_, conv2_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + SiLU for out2
        out2_ = torch.empty_like(out2, dtype=torch.float32)
        sums2 = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out2.device)
        sumsq2 = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out2.device)
        invstd2 = torch.empty(Bn * self.num_groups, dtype=torch.float32, device=out2.device)

        grid_sums2 = (Bn * self.num_groups,)
        groupnorm_sums_kernel[grid_sums2](
            out2, sums2, sumsq2,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )
        groupnorm_invstd_kernel[grid_sums2](
            sums2, sumsq2, invstd2,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )
        groupnorm_silu_apply_kernel[grid_sums2](
            out2, out2_, norm2_weight, norm2_bias, invstd2,
            Bn, Cn, Hn, Wn, self.num_groups, C_PER_GROUP,
        )

        # Residual add
        out = out2_ + x  # both float32

        return out


def run(*args):
    return ModelNew()(*args)
