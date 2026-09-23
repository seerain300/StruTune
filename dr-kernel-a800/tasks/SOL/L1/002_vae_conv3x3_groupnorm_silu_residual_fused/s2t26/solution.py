import torch
import triton
import triton.language as tl


def _validate_groupnorm_inputs(x: torch.Tensor, num_groups: int):
    B, C, H, W = x.shape
    if C % num_groups != 0:
        raise ValueError(f"num_groups={num_groups} must divide C={C}")
    return B, C, H, W, C // num_groups


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h = pid_h
    w = pid_w

    acc = tl.float32(0.0)

    # Loop over input channels
    for ci in range(0, C):
        # 3x3 neighborhood with padding=1 (masked via rh/rw checks)
        for dh in range(-1, 2):
            rh = h + dh
            valid_h = (rh >= 0) & (rh < H)
            for dw in range(-1, 2):
                rw = w + dw
                valid_w = (rw >= 0) & (rw < W)
                if valid_h & valid_w:
                    # Load input x[n, ci, rh, rw]
                    x_idx = ((pid_n * C + ci) * H + rh) * W + rw
                    x_val = tl.load(x_ptr + x_idx)
                    # Load weight w[co, ci, 1+dh, 1+dw] (layout: (C_OUT, C, 3, 3))
                    kh = 1 + dh
                    kw = 1 + dw
                    w_idx = ((pid_co * C + ci) * 3 * 3) + (kh * 3 + kw)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    # Store output out[n, co, h, w]
    out_idx = ((pid_n * C_OUT + pid_co) * H + h) * W + w
    tl.store(out_ptr + out_idx, acc)


# Triton kernel: compute sum and sum of squares per (n, group) over channels in group and all H*W elements
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    # Grid: (B * num_groups,)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    start_ci = g * C_PER_GROUP
    group_size = C_PER_GROUP * H * W

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + pid, s)
    tl.store(sumsq_ptr + pid, s2)


# Triton kernel: compute invstd per (n, group) using sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # (B * num_groups,)
    n = pid // num_groups
    g = pid % num_groups
    s = tl.load(sums_ptr + pid)
    s2 = tl.load(sumsq_ptr + pid)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    # Numerical stability: add small epsilon
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(invstd_ptr + pid, invstd)


# Triton kernel: apply GroupNorm + affine + SiLU per (n, group)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # (B * num_groups,)
    n = pid // num_groups
    g = pid % num_groups
    invstd = tl.load(invstd_ptr + pid)

    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)
                # GroupNorm normalize: y = (x - mean) * invstd
                # Note: we don't have per-element mean here; for GroupNorm, normalization is per (n, group),
                # so we apply with invstd computed for the group. This kernel applies per element assuming
                # mean is accounted for by invstd through precomputed stats; however, we need the mean explicitly.
                # Fix: We compute mean as s / (C_PER_GROUP * H * W) inside the kernel by reading s. We'll pass mean instead of invstd.
                # To keep clarity, we will instead launch a separate kernel that takes mean and invstd.
                # Placeholder: recompute mean per ci,h,w using sums; not efficient. Better: pass mean and invstd via buffers.
                # We will modify the invstd kernel to also store mean if needed. For simplicity and performance, we keep mean+invstd kernels.
                # Since Triton kernels can't easily return two values, we implement mean+invstd kernels separately.
                # This requires two kernels: sums and means+invstd, and a third apply kernel.
                # We already have sums and invstd kernels. We need mean. We will add a means_invstd kernel that computes mean and invstd.
                # (This note explains the design intent; see below for a revised implementation that includes mean).
                pass  # Placeholder — actual implementation below.


# Revised GroupNorm split: separate kernel to compute mean and invstd
@triton.jit
def groupnorm_mean_invstd_kernel(
    sums_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # (B * num_groups,)
    n = pid // num_groups
    g = pid % num_groups
    s = tl.load(sums_ptr + pid)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    # We don't have sumsq here; so we cannot compute var. We must also compute sumsq in the same kernel by storing it.
    # To avoid confusion, we will keep two-kernel approach: sums and sumsq, then mean_invstd that uses both.
    # But Triton can't return multiple outputs from a single launch. We'll implement two kernels: sums, sumsq, mean_invstd using both.
    # However, Triton requires us to write kernels as-is. So we'll add a second kernel that reads sums and sumsq to produce mean and invstd.
    # Since we cannot directly access sumsq from this kernel, we keep the two-kernel approach where groupnorm_sums_kernel writes sums and sumsq separately.


# Implement actual mean+invstd kernel reading sums and sumsq
@triton.jit
def groupnorm_mean_invstd_kernel_two(
    sums_ptr, sumsq_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # (B * num_groups,)
    n = pid // num_groups
    g = pid % num_groups
    s = tl.load(sums_ptr + pid)
    s2 = tl.load(sumsq_ptr + pid)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + pid, mean)
    tl.store(invstd_ptr + pid, invstd)


# Triton kernel: apply GroupNorm + affine + SiLU per (n, group) using mean and invstd
@triton.jit
def groupnorm_silu_apply_kernel_with_mean(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # (B * num_groups,)
    n = pid // num_groups
    g = pid % num_groups
    mean = tl.load(mean_ptr + pid)
    invstd = tl.load(invstd_ptr + pid)

    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)
                # GroupNorm normalize
                y = (x_val - mean) * invstd
                y = y * w + b  # affine
                # SiLU: y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


# Optional Triton elementwise add for residual (x + out). We keep PyTorch add here for simplicity.
# If strictly required Triton for residual, uncomment and call add_residual_kernel.
# def add_residual_kernel(out_ptr, x_ptr, B, C, H, W):
#     total = B * C * H * W
#     pid = tl.program_id(0)
#     offsets = pid * 1024 + tl.arange(0, 1024)
#     mask = offsets < total
#     for i in range(0, 1024):
#         idx = offsets[i]
#         if mask[i]:
#             x_val = tl.load(x_ptr + idx)
#             out_val = tl.load(out_ptr + idx)
#             tl.store(out_ptr + idx, out_val + x_val)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Triton-only implementation:
        - Conv1: Triton kernel computes out1 = conv(x, conv1_weight, no bias)
        - GroupNorm1 + SiLU: Triton kernels (sums, sumsq, mean+invstd, apply)
        - Conv2: Triton kernel computes out2 = conv(out1, conv2_weight, no bias)
        - GroupNorm2 + SiLU: Triton kernels (sums, sumsq, mean+invstd, apply)
        - Residual: out = out2 + x (PyTorch add)
        """
        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.float()
        if conv1_weight.dtype != torch.float32:
            conv1_weight = conv1_weight.float()
        if conv2_weight.dtype != torch.float32:
            conv2_weight = conv2_weight.float()
        if norm1_weight.dtype != torch.float32:
            norm1_weight = norm1_weight.float()
        if norm1_bias.dtype != torch.float32:
            norm1_bias = norm1_bias.float()
        if norm2_weight.dtype != torch.float32:
            norm2_weight = norm2_weight.float()
        if norm2_bias.dtype != torch.float32:
            norm2_bias = norm2_bias.float()

        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()  # shape (C_OUT, C, 3, 3)
        conv2_weight = conv2_weight.contiguous()  # shape (C_OUT, C, 3, 3)
        norm1_weight = norm1_weight.contiguous()  # shape (C,)
        norm1_bias = norm1_bias.contiguous()      # shape (C,)
        norm2_weight = norm2_weight.contiguous()  # shape (C,)
        norm2_bias = norm2_bias.contiguous()      # shape (C,)

        B, C, H, W = x.shape
        num_groups = 32
        C_PER_GROUP = C // num_groups
        _validate_groupnorm_inputs(x, num_groups)

        # Stage 1: Conv1
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nchw_kernel[grid_conv1](x, conv1_weight, out1, B, C, H, W, C)

        # Stage 1: GroupNorm + SiLU
        B_out, C_out, H_out, W_out = out1.shape  # (B, C, H, W)
        sums1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        grid_sums1 = (B * num_groups,)
        groupnorm_sums_kernel[grid_sums1](out1, sums1, sumsq1, B, C, H, W, num_groups, C_PER_GROUP)

        # Compute mean and invstd
        mean1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        invstd1 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        groupnorm_mean_invstd_kernel_two[grid_sums1](sums1, sumsq1, mean1, invstd1, B, C, H, W, num_groups, C_PER_GROUP)

        out1_norm = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_apply1 = (B * num_groups,)
        groupnorm_silu_apply_kernel_with_mean[grid_apply1](
            out1, out1_norm, norm1_weight, norm1_bias, mean1, invstd1, B, C, H, W, num_groups, C_PER_GROUP
        )

        # Stage 2: Conv2
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nchw_kernel[grid_conv2](out1_norm, conv2_weight, out2, B, C, H, W, C)

        # Stage 2: GroupNorm + SiLU
        sums2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        grid_sums2 = (B * num_groups,)
        groupnorm_sums_kernel[grid_sums2](out2, sums2, sumsq2, B, C, H, W, num_groups, C_PER_GROUP)

        mean2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        invstd2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        groupnorm_mean_invstd_kernel_two[grid_sums2](sums2, sumsq2, mean2, invstd2, B, C, H, W, num_groups, C_PER_GROUP)

        out2_norm = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_apply2 = (B * num_groups,)
        groupnorm_silu_apply_kernel_with_mean[grid_apply2](
            out2, out2_norm, norm2_weight, norm2_bias, mean2, invstd2, B, C, H, W, num_groups, C_PER_GROUP
        )

        # Residual connection: out = out2_norm + x
        # We keep PyTorch add for simplicity:
        out = out2_norm + x

        return out


def run(*args):
    return ModelNew()(*args)
