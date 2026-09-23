import torch
import triton
import triton.language as tl


# Triton kernel: Conv2d 3x3, stride=1, padding=1, NCHW layout, no bias
# Computes out[n, co, ho, wo] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, ho+dh, wo+dw] * w[co, ci, 1+dh, 1+dw]
# x, w, out are contiguous NCHW, float32.
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    KH: tl.constexpr, KW: tl.constexpr,  # 3x3
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,  # 1
    BLOCK_W: tl.constexpr,                 # vectorization across W
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    n = tl.program_id(0)
    co = tl.program_id(1)
    ho = tl.program_id(2)
    wb = tl.program_id(3)

    wo_start = wb * BLOCK_W
    wo = wo_start + tl.arange(0, BLOCK_W)
    mask_w = wo < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C):
        for dh in range(-PAD_H, PAD_H + 1):
            for dw in range(-PAD_W, PAD_W + 1):
                hi = ho + dh
                if hi < 0 or hi >= H:
                    continue
                wi = wo + dw
                valid = (wi >= 0) & (wi < W) & mask_w
                x_offset = ((n * C + ci) * H + hi) * W + wi
                x_vals = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                # w index: ((co * C) + ci) * (KH*KW) + (dh+1)*KW + (dw+1)
                w_idx = ((co * C) + ci) * (KH * KW) + (dh + 1) * KW + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_vals * w_val

    out_offset = ((n * C_OUT + co) * H + ho) * W + wo
    tl.store(out_ptr + out_offset, acc, mask=mask_w)


# Triton kernel: per (n, group) reduction for GroupNorm -> computes sum and sumsq
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


# Triton kernel: per (n, group) compute inverse std from sums and sumsq
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps hardcoded
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using precomputed invstd
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

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x = tl.load(x_ptr + idx)
                y = (x - mean) * invstd * w + b
                # SiLU: y * sigmoid(y)
                z = y * tl.sigmoid(y)
                tl.store(out_ptr + idx, z)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        # Epsilon for GroupNorm; the original uses eps argument, but we use a fixed 1e-5 here.
        self.eps = 1e-5

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Validate inputs
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "num_groups must divide C"
        # conv weights are (C_out, C_in, 3, 3) and here out_channels == C
        assert conv1_weight.shape[1] == C and conv1_weight.shape[0] == C, "conv1_weight must be (C, C, 3, 3)"
        assert conv2_weight.shape[1] == C and conv2_weight.shape[0] == C, "conv2_weight must be (C, C, 3, 3)"

        # Ensure contiguous and use float32 for compute
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        # Allocate output buffers for convs and GN stages
        out1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        C_OUT = C  # first conv: in_channels=C, out_channels=C

        # First conv in Triton: compute out1
        grid_conv1 = (B, C_OUT, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid_conv1](
            x_f32, conv1_weight.to(torch.float32), out1,
            B, C, H, W, C_OUT,
            KH=3, KW=3, PAD_H=1, PAD_W=1, BLOCK_W=64
        )

        # GroupNorm + SiLU for out1 using Triton (reduction + invstd + apply)
        HW = H * W
        C_per_group = C // self.num_groups
        n_groups = self.num_groups

        # reduction
        sums1 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        sumsq1 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        groupnorm_sums_kernel[(B * n_groups,)](out1, sums1, sumsq1, B, C, HW, n_groups, C_PER_GROUP=C_per_group)

        # invstd
        invstd1 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        groupnorm_invstd_kernel[(B * n_groups,)](sums1, sumsq1, invstd1, B, C, HW, n_groups, C_PER_GROUP=C_per_group)

        # apply: normalize + affine + SiLU
        out1_gn_silu = torch.empty_like(out1, dtype=torch.float32, device=x.device)
        groupnorm_silu_apply_kernel[(B * n_groups,)](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_gn_silu, invstd1,
            B, C, H, W, n_groups,
            C_PER_GROUP=C_per_group
        )

        # Second conv in Triton: compute out2
        out2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, C_OUT, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid_conv2](
            out1_gn_silu, conv2_weight.to(torch.float32), out2,
            B, C, H, W, C_OUT,
            KH=3, KW=3, PAD_H=1, PAD_W=1, BLOCK_W=64
        )

        # GroupNorm + SiLU for out2 using Triton (reduction + invstd + apply)
        sums2 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        sumsq2 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        groupnorm_sums_kernel[(B * n_groups,)](out2, sums2, sumsq2, B, C, HW, n_groups, C_PER_GROUP=C_per_GROUP)

        invstd2 = torch.empty(B * n_groups, dtype=torch.float32, device=x.device)
        groupnorm_invstd_kernel[(B * n_groups,)](sums2, sumsq2, invstd2, B, C, HW, n_groups, C_PER_GROUP=C_PER_GROUP)

        out2_gn_silu = torch.empty_like(out2, dtype=torch.float32, device=x.device)
        groupnorm_silu_apply_kernel[(B * n_groups,)](
            out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_gn_silu, invstd2,
            B, C, H, W, n_groups,
            C_PER_GROUP=C_PER_GROUP
        )

        # Residual add (PyTorch elementwise, minimal cost)
        out = out2_gn_silu + x_f32

        # Cast back to original dtype
        return out.to(x.dtype)


def run(*args):
    return ModelNew()(*args)
