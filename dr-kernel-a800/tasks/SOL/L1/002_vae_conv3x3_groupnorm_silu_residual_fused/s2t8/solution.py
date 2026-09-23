import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: compute per-(n, group) sum and sum of squares across all channels in the group and all spatial elements
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute mean and invstd per (n, group) from sums and sumsq; store mean and invstd to separate 1-element tensors
@triton.jit
def groupnorm_mean_invstd_kernel(
    sums_ptr, sumsq_ptr, mean_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    # store scalars to 1-element tensors
    tl.store(mean_ptr + out_idx, mean)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using per-(n,g) mean and invstd passed as pointers to 1-element tensors
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr,
    mean_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    mean = tl.load(mean_ptr + out_idx)
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        gamma = tl.load(norm_w_ptr + ci)
        beta = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                y = (x_val - mean) * invstd  # per-group normalization
                y = y * gamma + beta         # affine
                # SiLU: y = y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                y = y * sig
                tl.store(out_ptr + idx, y)


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci} sum_{dh in {-1,0,1}} sum_{dw in {-1,0,1}} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
# We pass H and W as tl.constexpr to allow Triton to compile the loops. This is acceptable for the provided workloads.
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B, C_out, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-PAD_H, PAD_H + KH):
            for dw in range(-PAD_W, PAD_W + KW):
                h_in = h + dh
                w_in = w_offsets + dw
                mask_hw = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
                # x layout: ((n * C_in + ci) * H + h_in) * W + w_in
                x_idx = ((pid_n * C_in + ci) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)
                # w layout: (C_out, C_in, 3, 3)
                w_index = pid_co * (C_in * 3 * 3) + ci * (3 * 3) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    out_index = ((pid_n * C_out + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_index, acc, mask=mask_w)


# ModelNew: entry point with Triton computation
class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure inputs are contiguous and float32 for Triton
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # Parameters
        B, C, H, W = x.shape
        C_in = C  # input channels = output channels of previous layer
        C_out = C
        _assert_divisible(C, self.num_groups)
        C_PER_GROUP = C // self.num_groups

        # Save residual
        residual = x

        # First stage: Conv3x3 -> GroupNorm -> SiLU
        # PyTorch conv
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + SiLU
        # Compute sums and sumsq per (n, group)
        sums = torch.empty(B * self.num_groups, device=out1.device, dtype=torch.float32)
        sumsq = torch.empty(B * self.num_groups, device=out1.device, dtype=torch.float32)
        groupnorm_sums_kernel[(B * self.num_groups,)](
            out1, sums, sumsq,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        # Compute mean and invstd
        mean = torch.empty(B * self.num_groups, device=out1.device, dtype=torch.float32)
        invstd = torch.empty(B * self.num_groups, device=out1.device, dtype=torch.float32)
        groupnorm_mean_invstd_kernel[(B * self.num_groups,)](
            sums, sumsq, mean, invstd,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        # Apply normalization + affine + SiLU
        out1_norm = torch.empty_like(out1)
        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            out1, norm1_weight, norm1_bias, out1_norm,
            mean, invstd,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        out = out1_norm

        # Second stage: Conv3x3 -> GroupNorm -> SiLU
        # PyTorch conv
        out2 = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + SiLU
        sums2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        sumsq2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        groupnorm_sums_kernel[(B * self.num_groups,)](
            out2, sums2, sumsq2,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        mean2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        invstd2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        groupnorm_mean_invstd_kernel[(B * self.num_groups,)](
            sums2, sumsq2, mean2, invstd2,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            out2, norm2_weight, norm2_bias, out2_norm,
            mean2, invstd2,
            N=B, C=C_out, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        # Residual connection
        out = out2_norm + residual

        # If original input dtype was not float32, cast back
        if x.dtype != torch.float32:
            out = out.to(x.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
