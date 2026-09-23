import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), row-major
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32), population std (unbiased=False)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Compute base offset for this (b, s) row
    base = (b * S + s) * F
    offs = base + tl.arange(0, BLOCK)
    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over features in BLOCK chunks
    for start in range(0, F, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = tl.float32(F)
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Store results
    out_idx = b * S + s
    tl.store(MEANS + out_idx, mean)
    tl.store(STDs + out_idx, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT[b,s,:] = max(0, X[b,s,:] - (MEANS[b*s] + STDs[b*s] * ICDF))
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base_in = (b * S + s) * F
    base_out = (b * S + s) * F

    mean = tl.load(MEANS + (b * S + s))
    std = tl.load(STDs + (b * S + s))
    cutoff = mean + std * tl.load(ICDF)  # ICDF is a scalar

    for start in range(0, F, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base_in + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(OUT + base_out + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a given scalar probability P (float32).
    Uses Abramowitz & Stegun 5.2.23 central region approximation:
    For p in [0.5, 1]: z = sqrt(2) * erfinv(2p - 1)
    In Triton, we implement a reasonable approximation without relying on erfinv.
    """
    # Load probability
    p = tl.load(P)  # scalar float32
    # Central region: p >= 0.5
    # a, b coefficients (central region)
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

    q = p - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly / den  # central region approximation
    tl.store(ICDF, z)


def _select_block(F: int) -> int:
    # Choose largest BLOCK from {4096, 2048, 1024} that does not exceed F.
    # This reduces loop iterations and improves throughput.
    if F >= 4096:
        return 4096
    if F >= 2048:
        return 2048
    return 1024


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes cutoff = mean + std * icdf(target_sparsity), then:
          y = max(0, x - cutoff), per (batch, seq, feature).
        All computations are done in Triton kernels.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Convert to float32 for computation (matching original behavior)
        x_fp32 = x.contiguous().to(torch.float32)

        B, S, F = x_fp32.shape
        device = x_fp32.device

        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=device)
        out = torch.empty((B, S, F), dtype=torch.float32, device=device)

        # Select BLOCK and num_warps
        BLOCK = _select_block(F)
        num_warps_stats = 8 if BLOCK >= 2048 else 4
        num_warps_sp = 8 if BLOCK >= 2048 else 4

        # Launch rowwise stats kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=device)
        p_dev = torch.empty((), dtype=torch.float32, device=device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return out.to(torch.bfloat16)