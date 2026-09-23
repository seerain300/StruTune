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

    # Base offset for the row
    base = b * S + s
    offset = 0
    sum_val = 0.0
    sum_sq = 0.0

    while offset < F:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        # Pointer to the row slice for this (b, s)
        ptr = X + base * F + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)
        offset += BLOCK

    mean = sum_val / F
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / F - mean * mean
    # Avoid tiny negative due to roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write out
    tl.store(MEANS + base, mean)
    tl.store(STDs + base, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a given probability P (scalar) using
    Abramowitz & Stegun 5.2.23 central-region approximation. Writes result to ICDF.
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5

    # Coefficients for central region
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

    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    num = poly * q
    icdf = num / den
    tl.store(ICDF, icdf)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply adaptive threshold: out = max(0, X - (MEANS[b*s] + STDs[b*s] * ICDF)).
    X: [B, S, F] input (float32)
    MEANS: [B*S] float32
    STDs: [B*S] float32
    ICDF: scalar float32
    OUT: [B, S, F] float32
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S + s
    mean = tl.load(MEANS + base)
    std = tl.load(STDs + base)
    scale = mean + std * tl.load(ICDF)  # ICDF is scalar

    offset = 0
    while offset < F:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        in_ptr = X + base * F + idx
        vals = tl.load(in_ptr, mask=mask, other=0.0)
        th = scale
        y = tl.maximum(vals - th, 0.0)  # ReLU
        out_ptr = OUT + base * F + idx
        tl.store(out_ptr, y, mask=mask)
        offset += BLOCK


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation:
    - Compute per-(batch, seq) row mean and std across features.
    - threshold = mean + std * icdf(target_sparsity) where icdf is inverse normal CDF.
    - Return max(0, inputs - threshold) in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and compute in float32
    x = inputs.contiguous()
    x_fp32 = x.to(torch.float32)

    B, S, F = x_fp32.shape

    # Allocate outputs
    out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

    # Per-row means and stds
    means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

    # Fixed BLOCK for robustness and previously validated performance
    BLOCK = 1024

    # Launch rowwise stats kernel
    grid = (B * S,)
    _rowwise_stats_kernel[grid](
        x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=4, num_stages=2
    )

    # Compute icdf for target_sparsity using Triton (single scalar)
    icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
    )

    # Launch sparsify + ReLU kernel
    _sparsify_relu_kernel[grid](
        x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=4, num_stages=2
    )

    # Return in bfloat16, matching original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)