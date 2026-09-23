import torch
import triton
import triton.language as tl


# Triton kernel: per (n, group) compute sum and sum of squares across channels in the group and all H*W elements.
# x: pointer to input tensor (N, C, H, W), float32 contiguous.
# sums_ptr[pid]: stores sum for this (n, group)
# sumsq_ptr[pid]: stores sum of squares for this (n, group)
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    # We'll compute sum and sumsq over all channels in group g and all spatial positions
    total_elems = C_PER_GROUP * H * W
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # x_ptr is float32
                s += x_val
                s2 += x_val * x_val

    out_idx = n * num_groups + g
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: per (n, group) compute mean and invstd from sums and sumsq.
# sums_ptr, sumsq_ptr: input per-(n,group) sums and sumsq
# invstd_ptr: output per-(n,group) invstd = 1/sqrt(var + eps), var = E[x^2] - mean^2
@triton.jit
def groupnorm_compute_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
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
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: per (n, group) apply normalization (using mean and invstd), affine (norm_weight, norm_bias), and SiLU activation.
# x_ptr: input tensor (N, C, H, W), float32
# out_ptr: output tensor (N, C, H, W), float32
# norm_w_ptr, norm_b_ptr: affine parameters (C,), float32
# mean_ptr, invstd_ptr: per-(n,group) mean and invstd, float32, length B*num_groups
@triton.jit
def groupnorm_apply_silu_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, mean_ptr, invstd_ptr,
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
                x_val = tl.load(x_ptr + idx)
                # normalize
                norm_val = (x_val - mean) * invstd
                # affine
                z = norm_val * scale + bias
                # SiLU activation: z * sigmoid(z) where sigmoid(z) = 1 / (1 + exp(-z))
                sig = 1.0 / (1.0 + tl.exp(-z))
                y = z * sig
                tl.store(out_ptr + idx, y)


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
    ):
        # Validate shapes
        B, C, H, W = x.shape
        _assert_divisible(C, 32)
        C_PER_GROUP = C // 32

        # Ensure contiguous and float32 for Triton kernels
        device = x.device
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # Stage 1: conv1
        y1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Stage 1: GroupNorm + SiLU
        Bn = B
        num_groups = 32
        grid_sums1 = (Bn * num_groups,)
        sums1 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        sumsq1 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums1](y1, sums1, sumsq1, Bn, C, H, W, num_groups, C_PER_GROUP)

        mean1 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        invstd1 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        groupnorm_compute_invstd_kernel[(Bn * num_groups,)](sums1, sumsq1, invstd1, Bn, C, H, W, num_groups, C_PER_GROUP, self.eps)

        out1_norm = torch.empty((Bn, C, H, W), device=device, dtype=torch.float32)
        groupnorm_apply_silu_kernel[(Bn * num_groups,)](
            y1, out1_norm, norm1_weight, norm1_bias, mean1, invstd1, Bn, C, H, W, num_groups, C_PER_GROUP
        )

        # Stage 2: conv2
        y2 = torch.nn.functional.conv2d(out1_norm, conv2_weight, bias=None, stride=1, padding=1)

        # Stage 2: GroupNorm + SiLU
        grid_sums2 = (Bn * num_groups,)
        sums2 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        sumsq2 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums2](y2, sums2, sumsq2, Bn, C, H, W, num_groups, C_PER_GROUP)

        mean2 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        invstd2 = torch.empty(Bn * num_groups, device=device, dtype=torch.float32)
        groupnorm_compute_invstd_kernel[(Bn * num_groups,)](sums2, sumsq2, invstd2, Bn, C, H, W, num_groups, C_PER_GROUP, self.eps)

        out2_norm = torch.empty((Bn, C, H, W), device=device, dtype=torch.float32)
        groupnorm_apply_silu_kernel[(Bn * num_groups,)](
            y2, out2_norm, norm2_weight, norm2_bias, mean2, invstd2, Bn, C, H, W, num_groups, C_PER_GROUP
        )

        # Residual add
        out = out2_norm + x  # x is float32

        return out


def run(*args):
    return ModelNew()(*args)
