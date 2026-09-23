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

    base = (b * S + s) * F

    sum_val = 0.0
    sum_sq = 0.0

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    var = sum_sq / F - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var)

    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a single scalar probability P using A&S 5.2.23 (central region).
    OUT: 1-element output tensor (float32)
    """
    p = tl.load(P)
    # Clamp to safe range; central region formula is valid for p in (0.5, 1)
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

    # Central region approximation (Abramowitz & Stegun 5.2.23)
    # q = p - 0.5
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

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    thr = poly / denom

    tl.store(OUT, thr)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification with ReLU:
    y = max(0, x - (mean + std * icdf))
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = (b * S + s) * F

    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    thr_mult = tl.load(ICDF)  # scalar icdf

    threshold = mean + std * thr_mult

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version:
        - Compute per-(batch, seq) mean and population std (unbiased=False).
        - Compute inverse normal CDF for target_sparsity using A&S 5.2.23 (central region) in Triton.
        - Apply sparsification: max(0, x - (mean + std * icdf)) and return bfloat16.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        B, S, F = x.shape

        # Compute in fp32 for stability
        x_fp32 = x.to(torch.float32)

        # Output buffer (fp32 for compute)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Choose BLOCK and num_warps based on F for better throughput
        BLOCK = 2048 if F >= 2048 else 1024
        num_warps_stats = 8 if BLOCK == 2048 else 4
        num_warps_sp = 8 if BLOCK == 2048 else 4

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=num_warps_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, num_warps=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=num_warps_sp
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)