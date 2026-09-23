import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias.
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
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

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-1, 2):
            h_curr = h + dh  # safe for h in [0,H-1]
            for dw in range(-1, 2):
                w_offsets_curr = w_offsets + dw
                mask_w_curr = (w_offsets_curr >= 0) & (w_offsets_curr < W)
                # Load weight scalar w[co, ci, dh+1, dw+1]
                w_idx = (pid_co * C * 9) + (ci * 9) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                # Load input vector x[n, ci, h_curr, w_offsets_curr]
                x_idx = ((pid_n * C + ci) * H + h_curr) * W + w_offsets_curr
                x_val = tl.load(x_ptr + x_idx, mask=mask_w_curr, other=0.0)
                acc += x_val * w_val

    # Store output row
    out_idx = ((pid_n * C_OUT + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


# Triton kernel: compute sums and sumsq per (n, group) for GroupNorm
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

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

    out_idx = n * num_groups + g
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute mean and invstd per (n, group) from sums and sumsq
@triton.jit
def groupnorm_compute_mean_invstd_kernel(
    sums_ptr, sumsq_ptr, mean_ptr, invstd_ptr,
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
    tl.store(mean_ptr + out_idx, mean)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: apply GroupNorm with affine and SiLU per (n, group), using mean and invstd
@triton.jit
def groupnorm_apply_affine_silu_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, mean_ptr, invstd_ptr,
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
        w = tl.load(norm_w_ptr + ci)  # float32
        b = tl.load(norm_b_ptr + ci)  # float32
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)  # float32
                # GroupNorm normalize: y = ((x - mean) * invstd) * w + b
                y = ((x_val - mean) * invstd) * w + b
                # SiLU: y * sigmoid(y) = y * (1 / (1 + exp(-y)))
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps=1e-5, block_w_conv=64):
        super().__init__()
        self.conv1_weight = conv1_weight  # (C, C, 3, 3)
        self.conv2_weight = conv2_weight  # (C, C, 3, 3)
        self.norm1_weight = norm1_weight  # (C,)
        self.norm1_bias = norm1_bias      # (C,)
        self.norm2_weight = norm2_weight  # (C,)
        self.norm2_bias = norm2_bias      # (C,)
        self.eps = eps                    # not used; kept for compatibility
        self.block_w_conv = block_w_conv  # tiling along W for conv

    def forward(self, x: torch.Tensor):
        # Validate shapes
        if x.dim() != 4:
            raise ValueError(f"Input x must be 4D (N, C, H, W), got shape {tuple(x.shape)}")
        N, C, H, W = x.shape
        C1, C2, KH, KW = self.conv1_weight.shape
        C3, C4, KH2, KW2 = self.conv2_weight.shape
        if C1 != C or C3 != C2:
            raise ValueError(f"Convolution weight channel mismatches: conv1_weight shape {self.conv1_weight.shape}, x C={C}")
        if C3 != C or C4 != C:
            raise ValueError(f"Conv2 weight channel mismatches: conv2_weight shape {self.conv2_weight.shape}, x C={C}")
        if self.norm1_weight.shape[0] != C or self.norm1_bias.shape[0] != C:
            raise ValueError(f"GroupNorm1 weight/bias must have shape (C,), got {tuple(self.norm1_weight.shape)} and {tuple(self.norm1_bias.shape)}")
        if self.norm2_weight.shape[0] != C or self.norm2_bias.shape[0] != C:
            raise ValueError(f"GroupNorm2 weight/bias must have shape (C,), got {tuple(self.norm2_weight.shape)} and {tuple(self.norm2_bias.shape)}")

        num_groups = 32
        if C % num_groups != 0:
            raise ValueError(f"num_groups={num_groups} must divide C={C}")
        Cpg = C // num_groups

        # Ensure contiguous and float32
        x_in = x.contiguous()
        if x_in.dtype != torch.float32:
            x_in = x_in.float()

        # First conv in Triton: out1 = conv3x3(x_in)
        out1 = torch.empty((N, C, H, W), dtype=torch.float32, device=x_in.device)
        w1_ptr = self.conv1_weight.contiguous().float()
        grid1 = (N, C, H, triton.cdiv(W, self.block_w_conv))
        conv3x3_nchw_kernel[grid1](
            x_in, w1_ptr, out1,
            N, C, H, W, C,
            BLOCK_W=self.block_w_conv,
        )

        # GroupNorm + SiLU for out1 using Triton
        sums1 = torch.empty(N * num_groups, dtype=torch.float32, device=x_in.device)
        sumsq1 = torch.empty(N * num_groups, dtype=torch.float32, device=x_in.device)
        groupnorm_sums_kernel[(N * num_groups,)](
            out1, sums1, sumsq1,
            N, C, H, W, num_groups,
            C_PER_GROUP=Cpg,
        )
        mean1 = torch.empty(N * num_groups, dtype=torch.float32, device=x_in.device)
        invstd1 = torch.empty(N * num_groups, dtype=torch.float32, device=x_in.device)
        group


def run(*args):
    return ModelNew()(*args)
