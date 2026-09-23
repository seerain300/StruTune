import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32)
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_start = (b * S + s) * F

    # Accumulate sum and sum of squares for this row
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over features in chunks of BLOCK
    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        x = tl.load(X + row_start + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = F
    mean = sum_x / n
    var = sum_x2 / n - mean * mean
    std = tl.sqrt(var)

    # Write per-row statistics
    idx = b * S + s
    tl.store(MEANS + idx, mean)
    tl.store(STDs + idx, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse standard normal CDF for scalar probability P using Abramowitz & Stegun 5.2.23 (central region).
    ICDF: output scalar (float32)
    """
    # Load probability
    p = tl.load(P)  # scalar
    # Central region: p in [0.5, 1]
    q = p - 0.5
    r = q * q
    # Coefficients for numerator and denominator
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as per A&S
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly * q / denom
    tl.store(ICDF, icdf)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = ReLU(X - (MEANS[b*s] + STDs[b*s] * ICDF)).
    Broadcasting MEANS and STDs over feature dimension.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_start_in = (b * S + s) * F
    row_start_out = row_start_in  # OUT has same shape

    mean = tl.load(MEANS + (b * S + s))
    std = tl.load(STDs + (b * S + s))
    threshold = mean + std * tl.load(ICDF)  # ICDF is scalar

    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        x = tl.load(X + row_start_in + offsets, mask=mask, other=0.0)
        y = tl.maximum(x - threshold, 0.0)  # ReLU
        tl.store(OUT + row_start_out + offsets, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-(batch, seq) mean and std across feature dim, then sparsifies
    by zeroing values below mean + std * icdf(target_sparsity).
    """
    if target_sparsity == 0.0:
        return inputs

    # Compute in float32 for numerical stability
    x = inputs
    x_fp32 = x.to(torch.float32)

    B, S, F = x_fp32.shape
    # Output buffer (float32 for compute)
    out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

    # Per-row means and stds
    means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Choose BLOCK and num_warps based on F
    if F >= 2048:
        BLOCK_STATS = 2048
        num_warps_stats = 8
        BLOCK_SP = 2048
        num_warps_sp = 8
    else:
        BLOCK_STATS = 1024
        num_warps_stats = 4
        BLOCK_SP = 1024
        num_warps_sp = 4

    # Launch rowwise stats kernel
    grid = (B * S,)
    _rowwise_stats_kernel[grid](
        x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
    )

    # Compute icdf for target_sparsity using Triton (single scalar)
    icdf = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
    )

    # Launch sparsify + ReLU kernel
    _sparsify_relu_kernel[grid](
        x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
    )

    # Return in bfloat16, matching original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)