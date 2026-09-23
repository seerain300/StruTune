import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, ho, wo] = sum over ci and 3x3 neighborhood of x[n, ci, hi, wi] * w[co, ci, dh+1, dw+1]
# We vectorize along W with BLOCK_W constexpr; grid covers B, C_out, H_out, blocks along W.
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_IN, H_IN, W_IN, C_OUT, H_OUT, W_OUT,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)   # batch index
    pid_co = tl.program_id(1)  # output channel index
    pid_ho = tl.program_id(2)  # output height index
    pid_w_blk = tl.program_id(3)  # block index along width

    ho = pid_ho
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_OUT

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C_IN):
        # Accumulate over 3x3 neighborhood; hi,wi are padded by hi=ho+dh, wi=wo+dw
        for dh in range(-1, 2):
            hi = ho + dh
            for dw in range(-1, 2):
                wi = w_offsets + dw
                # mask for valid (hi, wi) and valid width
                mask = (hi >= 0) & (hi < H_IN) & (wi >= 0) & (wi < W_IN) & mask_w
                # Load input values
                x_idx = ((pid_n * C_IN + ci) * H_IN + hi) * W_IN + wi
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                # Load corresponding weights for this (co, ci, dh+1, dw+1)
                w_idx = ((pid_co * C_IN) + ci) * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                # Accumulate
                acc += x_val * w_val

    # Store output
    out_idx = ((pid_n * C_OUT + pid_co) * H_OUT + ho) * W_OUT + w_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


# Triton kernel: compute per (n, group) sum and sum of squares for GroupNorm
@triton.jit
def _groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

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
def _groupnorm_invstd_kernel(
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
    eps = 1e-5
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per element for (n, group)
@triton.jit
def _groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr,
    B, C, H, W, num_groups,
    MEAN_ptr, INVSTD_ptr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # range [0, B * num_groups)
    n = pid // num_groups
    g = pid % num_groups

    mean = tl.load(MEAN_ptr + pid)
    invstd = tl.load(INVSTD_ptr + pid)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        scale = tl.load(norm_w_ptr + ci)
        bias = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                # Normalize: (x - mean) * invstd, then affine: *scale + bias
                gn = (x_val - mean) * invstd
                gn = gn * scale + bias
                # SiLU: y = gn * sigmoid(gn) = gn / (1 + exp(-gn))
                sig = 1.0 / (1.0 + tl.exp(-gn))
                y = gn * sig
                tl.store(out_ptr + idx, y)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        """
        Fused residual block implemented entirely in Triton kernels:
        - Conv3x3 (no bias) -> GroupNorm(num_groups=32, affine=True) -> SiLU
        - Repeat
        - Add original input residual
        """
        # Ensure dtype and contiguity; we'll compute in float32 for stability
        B, C, H, W = x.shape
        _assert_divisible(C, 32)
        device = x.device

        # Cast to float32 for Triton computation
        x1 = x.to(torch.float32).contiguous()
        # Conv1: (B, C, H, W) -> (B, C, H, W)
        out1 = torch.empty_like(x1, dtype=torch.float32, device=device)

        # Triton conv1
        C_IN = C  # input channels
        C_OUT = C  # output channels
        H_OUT = H
        W_OUT = W
        BLOCK_W = 64  # vectorize along width; tuneable
        grid = (B, C_OUT, H_OUT, triton.cdiv(W_OUT, BLOCK_W))
        conv3x3_nchw_kernel[grid](x1, conv1_weight.to(torch.float32), out1, B, C_IN, H, W, C_OUT, H_OUT, W_OUT, BLOCK_W=BLOCK_W)

        # GroupNorm + SiLU stage 1
        num_groups = 32
        CPG = C // num_groups  # channels per group
        sums = torch.empty(B * num_groups, device=device, dtype=torch.float32)
        sumsq = torch.empty(B * num_groups, device=device, dtype=torch.float32)
        _groupnorm_sums_kernel[(B * num_groups,)](out1, sums, sumsq, B, C, H, W, num_groups, C_PER_GROUP=CPG)
        invstd = 1.0 / torch.sqrt(sums / (CPG * H * W) - (sums / (CPG * H * W)) ** 2 + 1e-5)  # invstd per (n, group)
        # Allocate output for stage 1 normalized + SiLU
        out1_silu = torch.empty_like(out1, dtype=torch.float32, device=device)
        _groupnorm_silu_apply_kernel[(B * num_groups,)](
            out1, out1_silu, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B, C, H, W, num_groups, sums, invstd, C_PER_GROUP=CPG
        )

        # Conv2
        out2 = torch.empty_like(x1, dtype=torch.float32, device=device)
        conv3x3_nchw_kernel[grid](out1_silu, conv2_weight.to(torch.float32), out2, B, C_IN, H_OUT, W_OUT, C_OUT, H_OUT, W_OUT, BLOCK_W=BLOCK_W)

        # GroupNorm + SiLU stage 2
        sums2 = torch.empty(B * num_groups, device=device, dtype=torch.float32)
        sumsq2 = torch.empty(B * num_groups, device=device, dtype=torch.float32)
        _groupnorm_sums_kernel[(B * num_groups,)](out2, sums2, sumsq2, B, C, H, W, num_groups, C_PER_GROUP=CPG)
        invstd2 = 1.0 / torch.sqrt(sums2 / (CPG * H * W) - (sums2 / (CPG * H * W)) ** 2 + 1e-5)
        out2_silu = torch.empty_like(out2, dtype=torch.float32, device=device)
        _groupnorm_silu_apply_kernel[(B * num_groups,)](
            out2, out2_silu, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B, C, H, W, num_groups, sums2, invstd2, C_PER_GROUP=CPG
        )

        # Residual add (PyTorch, lightweight)
        out = out2_silu + x1

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            out = out.to(x.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
