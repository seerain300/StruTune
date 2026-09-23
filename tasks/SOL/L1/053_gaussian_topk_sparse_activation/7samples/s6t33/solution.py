import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), contiguous along last dim
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer offset for this (b, s) row
    row_offset = (b * S + s) * F

    # Accumulators for sum and sum of squares (float32)
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over features in chunks of BLOCK with masking
    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        # Compute linear addresses: X is contiguous with stride F along last dim
        ptrs = X + row_offset + offsets
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)

    # Compute mean and population std (unbiased=False)
    mean = sum_x / F
    var = sum_x2 / F - mean * mean
    # Ensure non-negative due to numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results to output arrays
    out_index = b * S + s
    MEANS[out_index] = mean
    STDs[out_index] = std


@triton.jit
def _icdf_ndtri_scalar_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for scalar probability P using Abramowitz & Stegun 5.2.23 central region approximation.
    OUT: 1-element tensor to store icdf
    """
    # Load probability
    p = tl.load(P)
    # Central region approximation
    q = p - 0.5
    r = q * q

    # Coefficients
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # must be negative
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly * q / den

    tl.store(OUT, icdf)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    For each (b, s) row, compute threshold = mean + std * ICDF,
    then write ReLU(X - threshold) into OUT.
    MEANS: [B*S] float32
    STDs: [B*S] float32
    ICDF: 1-element float32
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_offset = (b * S + s) * F

    mean = MEANS[b * S + s]
    std = STDs[b * S + s]
    threshold = mean + std * tl.load(ICDF)

    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F

        x_ptrs = X + row_offset + offsets
        out_ptrs = OUT + row_offset + offsets

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes per-(batch, seq) mean and std, then threshold = mean + std * icdf(target_sparsity),
        and returns ReLU(x - threshold) in bfloat16.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous float32 input
        x_fp32 = x.contiguous().to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Launch rowwise stats kernel: one program per (b, s)
        BLOCK_STATS = 1024
        grid_stats = (B * S,)
        _rowwise_stats_kernel[grid_stats](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=4, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton scalar kernel
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_scalar_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Output buffer
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Launch sparsify + ReLU kernel: one program per (b, s), iterate over features
        BLOCK_SP = 1024
        grid_sp = (B * S,)
        _sparsify_relu_kernel[grid_sp](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=4, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
