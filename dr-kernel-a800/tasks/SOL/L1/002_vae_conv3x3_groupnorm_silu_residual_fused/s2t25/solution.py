import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv2d 3x3, stride=1, padding=1, NCHW layout, no bias.
# Computes out[n, co, ho, wo] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, ho+dh, wo+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_OUT: tl.constexpr,
):
    # Grid: (B, C_OUT, H, W) one program per output element
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    n = pid_n
    co = pid_co
    ho = pid_h
    wo = pid_w

    # Accumulator for this output element
    acc = tl.float32(0.0)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C):
        for dh in range(-1, 2):
            h = ho + dh
            # mask for h within bounds (padding=1)
            valid_h = (h >= 0) & (h < H)
            for dw in range(-1, 2):
                w = wo + dw
                valid_w = (w >= 0) & (w < W)
                valid = valid_h & valid_w

                # compute input and weight offsets
                # Input offset for NCHW: (((n * C + ci) * H + h) * W + w)
                x_off = ((n * C + ci) * H + h) * W + w
                # Weight offset for (co, ci, dh+1, dw+1): (((co * C + ci) * 3 + (dh + 1)) * 3 + (dw + 1))
                # Note: weight layout is (C_OUT, C, 3, 3)
                w_off = ((co * C + ci) * 9 + (dh + 1 + 0) * 3 + (dw + 1))  # constants 3,3

                # If valid, load and accumulate; else skip (we could also use other=0 in tl.load, but we prefer explicit)
                if valid:
                    x_val = tl.load(x_ptr + x_off)
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # Store result
    out_off = ((n * C_OUT + co) * H + ho) * W + wo
    tl.store(out_ptr + out_off, acc)


# Triton kernel: compute per-(n, group) sum and sum of squares across channels and spatial positions.
# x is NCHW contiguous float32
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    total = C_PER_GROUP * H * W

    # Iterate over channels in the group and all spatial positions
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
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    total = C_PER_GROUP * H * W
    mean = s / total
    var = s2 / total - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon from host
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply normalization and affine per (n, group), then SiLU
# x, norm_weight, norm_bias are NCHW contiguous float32, out is NCHW float32
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr, mean_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    invstd = tl.load(invstd_ptr + out_idx)
    mean = tl.load(mean_ptr + out_idx)

    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                base = (n * C + ci) * H + h
                idx = base * W + w_idx
                x_val = tl.load(x_ptr + idx)
                # Normalize and affine: y = ((x - mean) * invstd) * w + b
                y = (x_val - mean) * invstd
                y = y * w + b
                # SiLU: y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        B, C, H, W = x.shape
        _assert_divisible(C, self.num_groups)

        # Ensure contiguous and float32 for Triton
        x = x.contiguous().float()
        conv1_weight = conv1_weight.contiguous().float()
        conv2_weight = conv2_weight.contiguous().float()

        # First stage: Triton Conv3x3
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid = (B, C, H, W)
        conv3x3_nchw_kernel[grid](x, conv1_weight, out1, B=B, C=C, H=H, W=W, C_OUT=C)

        # GroupNorm and SiLU for first stage using Triton: compute sums and invstd in host, then apply in Triton
        # 1) Compute sums and sumsq per (n,group)
        sums = torch.zeros(B * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq = torch.zeros(B * self.num_groups, device=x.device, dtype=torch.float32)
        grid_sums = (B * self.num_groups,)
        C_PER_GROUP = C // self.num_groups
        groupnorm_sums_kernel[grid_sums](out1, sums, sumsq, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP)

        # 2) Compute mean and invstd per (n,group) on host (PyTorch ops are fine here)
        group_size = C_PER_GROUP * H * W
        mean = sums / group_size
        var = sumsq / group_size - mean * mean
        invstd = 1.0 / torch.sqrt(var + self.eps)
        # Save mean for apply kernel

        # 3) Apply normalization + affine + SiLU in Triton
        out1_gn_silu = torch.empty_like(out1)
        mean_ptr = mean  # we can pass torch tensor as pointer; Triton will read its values
        grid_apply = (B * self.num_groups,)
        groupnorm_silu_apply_kernel[grid_apply](
            out1, norm1_weight, norm1_bias, out1_gn_silu, invstd, mean,
            B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        # Add residual x
        y = out1_gn_silu + x

        # Second stage: Triton Conv3x3
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        conv3x3_nchw_kernel[grid](y, conv2_weight, out2, B=B, C=C, H=H, W=W, C_OUT=C)

        # GroupNorm and SiLU for second stage using Triton
        sums2 = torch.zeros(B * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq2 = torch.zeros(B * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums](out2, sums2, sumsq2, B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP)

        invstd2 = 1.0 / torch.sqrt(sums2 / group_size + sumsq2 / group_size / group_size - (sums2 / group_size) * (sums2 / group_size) + self.eps)
        # Compute mean2
        mean2 = sums2 / group_size

        out2_gn_silu = torch.empty_like(out2)
        grid_apply2 = (B * self.num_groups,)
        groupnorm_silu_apply_kernel[grid_apply2](
            out2, norm2_weight, norm2_bias, out2_gn_silu, invstd2, mean2,
            B=B, C=C, H=H, W=W, num_groups=self.num_groups, C_PER_GROUP=C_PER_GROUP
        )

        # Final residual add
        out = out2_gn_silu + x

        return out


def run(*args):
    return ModelNew()(*args)
