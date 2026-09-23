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

    # Base offset for this row
    base = (b * S + s) * F

    sum_val = 0.0
    sum_sq = 0.0

    # Loop over feature dimension in blocks
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    # population std, unbiased=False
    var = sum_sq / F - mean * mean
    var = tl.maximum(var, 0.0)  # clamp for numerical safety
    std = tl.sqrt(var)

    # Store results for this row
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a single scalar probability P using A&S 5.2.23 central region approximation.
    OUT: 1-element output tensor (float32)
    """
    p = tl.load(P)
    # Clamp to safe range
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

    # Constants for A&S 5.2.23 (central region)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.38357751867269e+02
    a5 = -3.066479806614716e+01  # negative as per original formula
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    q = p - 0.5
    r = q * q

    # Horner's method for numerator and denominator
    num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    out_val = num / den
    tl.store(OUT, out_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification per row: y = max(0, x - (mean + std * ICDF)),
    where ICDF is a 1-element tensor (scalar).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    threshold = tl.load(MEANS + pid) + tl.load(STDs + pid) * tl.load(ICDF)

    base = (b * S + s) * F

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
        Triton-optimized version of run:
        - Compute per-(batch, seq) mean and population std (unbiased=False).
        - Compute inverse normal CDF for target_sparsity using A&S 5.2.23 central region approximation in Triton.
        - Apply sparsification: max(0, x - (mean + std * icdf)) and return bfloat16.
        """
        x = x.contiguous()
        B, S, F = x.shape

        # Compute in fp32 for stability
        x_fp32 = x.to(torch.float32)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Launch rowwise stats kernel with fixed BLOCK to ensure robustness
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=1024, num_warps=4, num_stages=2
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
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=1024, num_warps=4, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)