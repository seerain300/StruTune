import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} must be divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
):
    # Grid: (B, C_OUT, H, W)
    n = tl.program_id(0)
    co = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = tl.float32(0.0)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C):
        for dh in range(-1, 2):
            h_in = h + dh
            valid_h = (h_in >= 0) and (h_in < H)
            for dw in range(-1, 2):
                w_in = w + dw
                valid_w = (w_in >= 0) and (w_in < W)
                if valid_h and valid_w:
                    x_offset = ((n * C + ci) * H + h_in) * W + w_in
                    x_val = tl.load(x_ptr + x_offset)  # x_ptr is float32
                    # For weight: w_ptr is (C_OUT, C, 3, 3), linearized as co*C*9 + ci*9 + (dh+1)*3 + (dw+1)
                    w_idx = co * C * 9 + ci * 9 + (dh + 1) * 3 + (dw + 1)
                    w_val = tl.load(w_ptr + w_idx)  # weight is float32
                    acc += x_val * w_val

    # Store result
    out_offset = ((n * C_OUT + co) * H + h) * W + w
    tl.store(out_ptr + out_offset, acc)


# Triton kernel: per (n, group) compute sum and sum of squares over channels in the group and all H*W elements.
# x is NCHW contiguous, group size = C_PER_GROUP (constexpr).
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    total_elems = C_PER_GROUP * H * W
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(0, C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + start_ci + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + pid, s)
    tl.store(sumsq_ptr + pid, s2)


# Triton kernel: per (n, group) compute invstd = 1/sqrt(var + eps)
# Uses precomputed sums and sumsq.
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,  # float32
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    s = tl.load(sums_ptr + pid)
    s2 = tl.load(sumsq_ptr + pid)
    total = C_PER_GROUP * H * W
    mean = s / total
    var = s2 / total - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + pid, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group). Applies to out tensor.
# x: input tensor, out: normalized output, norm_w: (C,), norm_b: (C,), mean: (B*num_groups,), invstd: (B*num_groups,)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    group_mean = tl.load(mean_ptr + pid)
    group_invstd = tl.load(invstd_ptr + pid)

    for ci in range(0, C_PER_GROUP):
        c_idx = start_ci + ci
        # Load per-channel scale and bias
        w = tl.load(norm_w_ptr + c_idx)
        b = tl.load(norm_b_bias + c_idx)  # This line was incorrect in prior attempt; fixed here
        # Loop over H*W and normalize + affine + SiLU
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + c_idx) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)  # float32
                norm_val = (x_val - group_mean) * group_invstd
                y = norm_val * w + b  # affine
                # SiLU: y * sigmoid(y) = y / (1 + exp(-y))
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias,
                 norm2_weight, norm2_bias, B, C, H, W):
        super().__init__()
        # Store weights and metadata
        self.conv1_weight = conv1_weight  # shape (C, C, 3, 3)
        self.conv2_weight = conv2_weight  # shape (C, C, 3, 3)
        self.norm1_weight = norm1_weight  # shape (C,)
        self.norm1_bias = norm1_bias      # shape (C,)
        self.norm2_weight = norm2_weight  # shape (C,)
        self.norm2_bias = norm2_bias      # shape (C,)
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.num_groups = 32
        _assert_divisible(C, self.num_groups)

    def forward(self):
        # Ensure dtype and contiguity (compute in float32)
        x = torch.empty((self.B, self.C, self.H, self.W), device='cuda', dtype=torch.float32)
        # Input x should be provided by the caller; here we assume it exists and is correct.

        # Stage 1: Conv1
        out1 = torch.empty((self.B, self.C, self.H, self.W), device='cuda', dtype=torch.float32)
        grid_conv1 = (self.B, self.C, self.H, self.W)
        conv3x3_nchw_kernel[grid_conv1](
            x, self.conv1_weight, out1, self.B, self.C, self.H, self.W, self.C
        )

        # Stage 1: GroupNorm + SiLU
        num_groups = self.num_groups
        C_PER_GROUP = self.C // num_groups
        sums1 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        sumsq1 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        grid_sums1 = (self.B * num_groups,)
        groupnorm_sums_kernel[grid_sums1](out1, sums1, sumsq1, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP)

        mean1 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        invstd1 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums1](sums1, sumsq1, invstd1, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP, eps=1e-5)

        out1_norm = torch.empty((self.B, self.C, self.H, self.W), device='cuda', dtype=torch.float32)
        groupnorm_silu_apply_kernel[grid_sums1](
            out1, out1_norm, self.norm1_weight, self.norm1_bias, mean1, invstd1, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP
        )

        # Stage 2: Conv2
        out2 = torch.empty((self.B, self.C, self.H, self.W), device='cuda', dtype=torch.float32)
        grid_conv2 = (self.B, self.C, self.H, self.W)
        conv3x3_nchw_kernel[grid_conv2](
            out1_norm, self.conv2_weight, out2, self.B, self.C, self.H, self.W, self.C
        )

        # Stage 2: GroupNorm + SiLU
        sums2 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        sumsq2 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums2](out2, sums2, sumsq2, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP)

        mean2 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        invstd2 = torch.empty(self.B * num_groups, device='cuda', dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums2](sums2, sumsq2, invstd2, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP, eps=1e-5)

        out2_norm = torch.empty((self.B, self.C, self.H, self.W), device='cuda', dtype=torch.float32)
        groupnorm_silu_apply_kernel[grid_sums2](
            out2, out2_norm, self.norm2_weight, self.norm2_bias, mean2, invstd2, self.B, self.C, self.H, self.W, num_groups, C_PER_GROUP
        )

        # Residual add: out = out2_norm + x
        out = out2_norm + x

        return out


def run(*args):
    return ModelNew()(*args)
