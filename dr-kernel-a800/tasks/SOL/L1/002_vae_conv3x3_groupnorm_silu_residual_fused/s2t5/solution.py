import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} must be divisible by {divisor}")


# Triton kernel: compute per (n, group) sums and sum of squares for GroupNorm
# x: (B, C, H, W) contiguous, float32
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    # Iterate channels in this group and all spatial positions
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std from sums and sumsq for each (n, group)
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon for numerical stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply GroupNorm (using mean and invstd), affine (norm_weight, norm_bias), and SiLU
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, mean_ptr, invstd_ptr, norm_w_ptr, norm_b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    mean = tl.load(mean_ptr + out_idx)    # scalar float32
    invstd = tl.load(invstd_ptr + out_idx)  # scalar float32

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        gamma = tl.load(norm_w_ptr + ci)  # scale
        beta = tl.load(norm_b_ptr + ci)   # bias
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                # Normalize
                y = (x_val - mean) * invstd * gamma + beta  # GroupNorm affine
                # SiLU activation: y = y * sigmoid(y) = y / (1 + exp(-y))
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float = 1e-5):
        # Validate shapes
        B, C, H, W = x.shape
        _assert_divisible(C, self.num_groups)

        # First path: Conv3x3 (use PyTorch/cuDNN) -> GroupNorm -> SiLU
        x1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # Triton GroupNorm + SiLU
        C_PER_GROUP = C // self.num_groups
        x1_f = x1.contiguous().to(torch.float32)

        sums1 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[(B * self.num_groups,)](
            x1_f, sums1, sumsq1, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        invstd1 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[(B * self.num_groups,)](
            sums1, sumsq1, invstd1, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        group_size = C_PER_GROUP * H * W
        mean1 = sums1 / group_size  # shape (B*num_groups,)

        out1_silu = torch.empty_like(x1_f)
        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            x1_f, mean1, invstd1, norm1_weight, norm1_bias, out1_silu,
            B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        # Second path: Conv3x3 (PyTorch) -> GroupNorm -> SiLU
        x2 = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)
        x2_f = x2.contiguous().to(torch.float32)

        sums2 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[(B * self.num_groups,)](
            x2_f, sums2, sumsq2, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        invstd2 = torch.empty(B * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[(B * self.num_groups,)](
            sums2, sumsq2, invstd2, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        mean2 = sums2 / group_size  # shape (B*num_groups,)

        out2_silu = torch.empty_like(x2_f)
        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            x2_f, mean2, invstd2, norm2_weight, norm2_bias, out2_silu,
            B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        # Residual connection: add original x (cast to float32 for consistency)
        residual = x.to(torch.float32)
        final = out2_silu + residual

        return final


def run(*args):
    return ModelNew()(*args)
