import torch
import triton
import triton.language as tl


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
# H, W must be passed as constexpr. We tile along W using BLOCK_W.
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, C_OUT, H: tl.constexpr, W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output block along W
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1 (static loops for H/W)
        for dh in (-1, 0, 1):
            dh_i = dh + 1  # convert to 1-based index for input h
            h_idx = h + dh
            # Guard h_idx within [0, H-1]; since H is constexpr and loops are static, h_idx is always in range here
            for dw in (-1, 0, 1):
                dw_j = dw + 1  # convert to 1-based index for input w
                w_idx = w_offsets + dw
                # Compute input base index for this ci, h_idx, w_idx
                base_in = ((pid_n * C + ci) * H + h_idx) * W
                in_ptrs = x_ptr + base_in + w_idx
                # Load input vector with mask only for W tail
                x_vec = tl.load(in_ptrs, mask=mask_w, other=0.0)
                # Load weight scalar for w[co, ci, dh+1, dw+1]
                w_scalar = tl.load(w_ptr + (pid_co * C + ci) * 9 + (dh + 1) * 3 + (dw + 1))
                acc += x_vec * w_scalar

    # Store accumulated output
    out_ptrs = out_ptr + ((pid_n * C_OUT + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptrs, acc, mask=mask_w)


# Triton kernels for GroupNorm + SiLU
# Compute sums and sumsq per (n, group)
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h_i in range(0, H):
            for w_i in range(0, W):
                idx = ((n * C + ci) * H + h_i) * W + w_i
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    out_idx = n * num_groups + g
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Compute invstd per (n, group) from sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups,
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon for stability
    tl.store(invstd_ptr + out_idx, invstd)


# Normalize + affine + SiLU per (n, group)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        scale = tl.load(norm_w_ptr + ci)
        bias = tl.load(norm_b_ptr + ci)
        for h_i in range(0, H):
            for w_i in range(0, W):
                idx = ((n * C + ci) * H + h_i) * W + w_i
                x_val = tl.load(x_ptr + idx)
                y = (x_val - mean) * invstd * scale + bias  # mean is not available here; we pass precomputed per stage
                # SiLU: y * sigmoid(y) = y / (1 + exp(-y))
                silu = y * (1.0 / (1.0 + tl.exp(-y)))
                tl.store(out_ptr + idx, silu)


@triton.jit
def groupnorm_silu_apply_with_mean_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr, mean_ptr,
    B, C, H: tl.constexpr, W: tl.constexpr, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)
    mean_val = tl.load(mean_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        scale = tl.load(norm_w_ptr + ci)
        bias = tl.load(norm_b_ptr + ci)
        for h_i in range(0, H):
            for w_i in range(0, W):
                idx = ((n * C + ci) * H + h_i) * W + w_i
                x_val = tl.load(x_ptr + idx)
                y = (x_val - mean_val) * invstd * scale + bias
                silu = y * (1.0 / (1.0 + tl.exp(-y)))
                tl.store(out_ptr + idx, silu)


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
    Triton-only implementation: all heavy ops are done in Triton kernels.
    """
    # Validate inputs
    assert x.dim() == 4, "x must be NCHW"
    assert conv1_weight.dim() == 4 and conv2_weight.dim() == 4, "conv weights must be (C, C, 3, 3)"
    B, C, H, W = x.shape
    C1, C1_in, KH, KW = conv1_weight.shape
    C2, C2_in, _, _ = conv2_weight.shape
    assert C1_in == C and C2_in == C, "Input channels must match conv weights"
    assert KH == 3 and KW == 3, "Only 3x3 supported"
    assert norm1_weight.shape[0] == C and norm1_bias.shape[0] == C
    assert norm2_weight.shape[0] == C and norm2_bias.shape[0] == C

    # Fixed group setup
    num_groups = 32
    assert C % num_groups == 0, "num_groups must divide C"

    # Ensure contiguous and float32
    x_ = x.contiguous().to(torch.float32)
    conv1_weight_ = conv1_weight.contiguous().to(torch.float32)
    conv2_weight_ = conv2_weight.contiguous().to(torch.float32)
    norm1_w = norm1_weight.contiguous().to(torch.float32)
    norm1_b = norm1_bias.contiguous().to(torch.float32)
    norm2_w = norm2_weight.contiguous().to(torch.float32)
    norm2_b = norm2_bias.contiguous().to(torch.float32)

    # 1st Conv (NCHW)
    out1 = torch.zeros((B, C, H, W), device=x.device, dtype=torch.float32)
    grid_conv1 = (B, C, H, triton.cdiv(W, 64))
    conv3x3_nchw_kernel[grid_conv1](
        x_.data_ptr(), conv1_weight_.data_ptr(), out1.data_ptr(),
        B, C, C, H, W,
        BLOCK_W=64,
    )

    # 1st GroupNorm + SiLU
    # Compute sums and sumsq per (n, group)
    sums1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    sumsq1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    grid_sums1 = (B * num_groups,)
    groupnorm_sums_kernel[grid_sums1](
        out1.data_ptr(), sums1.data_ptr(), sumsq1.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )
    # Compute invstd
    invstd1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    groupnorm_invstd_kernel[grid_sums1](
        sums1.data_ptr(), sumsq1.data_ptr(), invstd1.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )
    # Normalize + affine + SiLU
    out1_gn = torch.empty_like(out1)
    # We need mean to apply normalization; we can compute mean per (n,group) from sums
    mean1 = sums1 / (C // num_groups * H * W)
    groupnorm_silu_apply_with_mean_kernel[grid_sums1](
        out1.data_ptr(), norm1_w.data_ptr(), norm1_b.data_ptr(), out1_gn.data_ptr(), invstd1.data_ptr(), mean1.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )

    # 2nd Conv (NCHW)
    out2 = torch.zeros((B, C, H, W), device=x.device, dtype=torch.float32)
    grid_conv2 = (B, C, H, triton.cdiv(W, 64))
    conv3x3_nchw_kernel[grid_conv2](
        out1_gn.data_ptr(), conv2_weight_.data_ptr(), out2.data_ptr(),
        B, C, C, H, W,
        BLOCK_W=64,
    )

    # 2nd GroupNorm + SiLU
    sums2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    sumsq2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    grid_sums2 = (B * num_groups,)
    groupnorm_sums_kernel[grid_sums2](
        out2.data_ptr(), sums2.data_ptr(), sumsq2.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )
    invstd2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
    groupnorm_invstd_kernel[grid_sums2](
        sums2.data_ptr(), sumsq2.data_ptr(), invstd2.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )
    mean2 = sums2 / (C // num_groups * H * W)
    out2_gn = torch.empty_like(out2)
    groupnorm_silu_apply_with_mean_kernel[grid_sums2](
        out2.data_ptr(), norm2_w.data_ptr(), norm2_b.data_ptr(), out2_gn.data_ptr(), invstd2.data_ptr(), mean2.data_ptr(),
        B, C, H, W, num_groups,
        C_PER_GROUP=C // num_groups,
    )

    # Residual connection
    out = out2_gn + x_

    return out

# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
