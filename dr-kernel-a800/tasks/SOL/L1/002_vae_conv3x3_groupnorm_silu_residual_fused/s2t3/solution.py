import torch
import torch.nn as nn
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, ho, wo] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, ho+dh, wo+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,  # tile size along W
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    ho = pid_h
    wo_start = pid_w_blk * BLOCK_W
    wo_offsets = wo_start + tl.arange(0, BLOCK_W)
    mask_w = wo_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        # kh, kw in {-1, 0, 1}
        for kh in range(-1, 2):
            dh = ho + kh
            in_bounds_h = (dh >= 0) and (dh < H)
            if in_bounds_h:
                for kw in range(-1, 2):
                    dw = wo_offsets + kw
                    in_bounds_w = mask_w & (dw >= 0) & (dw < W)
                    # Build input indices: x[n, ci, dh, dw]
                    in_idx = ((pid_n * C + ci) * H + dh) * W + dw
                    # Build weight indices: w[co, ci, 1+kh, 1+kw]
                    w_idx = ((pid_co * C + ci) * 3 + (1 + kh)) * 3 + (1 + kw)
                    # Load with mask; default 0.0 for out-of-bounds
                    x_val = tl.load(x_ptr + in_idx, mask=in_bounds_w, other=0.0).to(tl.float32)
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                    acc += x_val * w_val

    # Store results
    out_idx = ((pid_n * C_OUT + pid_co) * H + ho) * W + wo_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


# Triton kernel: per-(n, group) compute sum and sumsq across channels and spatial
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
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
                x_val = tl.load(x_ptr + idx).to(tl.float32)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute invstd from sums and sumsq
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply normalization + affine + SiLU per element
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr, mean_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups)
    invstd = tl.load(invstd_ptr + pid)
    mean_val = tl.load(mean_ptr + pid)

    # Unpack pid into (n, g)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        gamma = tl.load(norm_w_ptr + ci).to(tl.float32)
        beta = tl.load(norm_b_ptr + ci).to(tl.float32)

        for h in range(0, H):
            for w in range(0, W):
                idx_in = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx_in).to(tl.float32)
                y = (x_val - mean_val) * invstd * gamma + beta
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                idx_out = ((n * C + ci) * H + h) * W + w
                tl.store(out_ptr + idx_out, out_val)


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Validate shapes and constraints
        B, C, H, W = x.shape
        if C != conv1_weight.shape[1] or conv1_weight.shape[0] != C:
            raise ValueError(f"conv1_weight shape {conv1_weight.shape} incompatible with input C={C}")
        if C != conv2_weight.shape[1] or conv2_weight.shape[0] != C:
            raise ValueError(f"conv2_weight shape {conv2_weight.shape} incompatible with input C={C}")
        if C % self.num_groups != 0:
            raise ValueError(f"num_groups={self.num_groups} must divide C={C}")
        _assert_divisible(C, self.num_groups)
        C_PER_GROUP = C // self.num_groups

        # Ensure contiguity and float32 compute
        x = x.contiguous()
        # First convolution: Triton
        out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Launch Triton conv1
        # Grid over (B, C, H, ceil_div(W, BLOCK_W))
        BLOCK_W = 32  # tuneable
        grid_conv = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv](
            x, conv1_weight, out,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
        )

        # GroupNorm + SiLU (first stage) in Triton
        sums = torch.empty(B * self.num_groups, device=out.device, dtype=torch.float32)
        sumsq = torch.empty(B * self.num_groups, device=out.device, dtype=torch.float32)
        invstd = torch.empty(B * self.num_groups, device=out.device, dtype=torch.float32)
        mean = torch.empty(B * self.num_groups, device=out.device, dtype=torch.float32)

        groupnorm_sums_kernel[(B * self.num_groups,)](
            out, sums, sumsq,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        groupnorm_invstd_kernel[(B * self.num_groups,)](
            sums, sumsq, invstd,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        mean = (sums / (C_PER_GROUP * H * W)).to(torch.float32)

        out1 = torch.empty_like(out)

        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            out, norm1_weight, norm1_bias, out1, invstd, mean,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Residual add
        out1 = out1 + x

        # Second convolution: Triton
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        grid_conv2 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv2](
            out1, conv2_weight, out2,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
        )

        # GroupNorm + SiLU (second stage) in Triton
        sums2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        sumsq2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        invstd2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)
        mean2 = torch.empty(B * self.num_groups, device=out2.device, dtype=torch.float32)

        groupnorm_sums_kernel[(B * self.num_groups,)](
            out2, sums2, sumsq2,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        groupnorm_invstd_kernel[(B * self.num_groups,)](
            sums2, sumsq2, invstd2,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        mean2 = (sums2 / (C_PER_GROUP * H * W)).to(torch.float32)

        out_final = torch.empty_like(out2)

        groupnorm_silu_apply_kernel[(B * self.num_groups,)](
            out2, norm2_weight, norm2_bias, out_final, invstd2, mean2,
            B, C, H, W, self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Final residual add
        out_final = out_final + x

        return out_final


def run(*args):
    return ModelNew()(*args)
