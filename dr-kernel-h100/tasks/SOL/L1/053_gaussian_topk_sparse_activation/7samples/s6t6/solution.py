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

    # Compute base offsets for this row
    base = (b * S + s) * F  # row index in flattened [B*S, F]

    sum_val = 0.0
    sum_sq = 0.0

    # Loop over feature dimension in blocks
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        # Load a block of the row
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    # population std, unbiased=False
    var = sum_sq / F - mean * mean
    # Avoid negative due to rounding: clamp to 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results for this row
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, B, S, F, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a single scalar probability P using A&S 5.2.23 approximation.
    OUT: 1-element output tensor (float32)
    """
    # Load scalar probability
    p = tl.load(P)
    # Clamp to safe range
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

    # Constants for A&S 5.2.23
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region: x < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    y_low = poly_low / denom_low

    # Upper region: x > p_high
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    denom_up = (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
    y_up = -poly_up / denom_up

    # Central region: p_low <= p <= p_high
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_mid_num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_mid_den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    y_mid = poly_mid_num * q_mid / poly_mid_den

    # Select region based on p without Python if (use tl.where)
    use_low = p < p_low
    use_up = p > p_high
    use_mid = ~use_low & ~use_up

    y = tl.zeros((), dtype=tl.float32)
    y = tl.where(use_low, y_low, y)
    y = tl.where(use_up, y_up, y)
    y = tl.where(use_mid, y_mid, y)

    tl.store(OUT, y)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = max(0, X - (MEANS[b*s] + STDs[b*s] * ICDF))
    X, OUT: [B, S, F] float32
    MEANS, STDs: [B*S] float32
    ICDF: 1-element tensor (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Load mean and std for this row
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    thr_mult = tl.load(ICDF)  # scalar inverse CDF

    # Compute threshold scalar for this row
    threshold = mean + std * thr_mult

    base = (b * S + s) * F

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        y = x - threshold  # elementwise subtract
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(OUT + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version of run:
        - Compute per-(batch, seq) mean and std (population).
        - Compute inverse normal CDF for target_sparsity using A&S 5.2.23 in Triton.
        - Apply sparsification: max(0, x - (mean + std * icdf)) and return bfloat16.
        """
        # Ensure input is contiguous and float32 for numerical stability
        x = x.contiguous()
        B, S, F = x.shape
        x_fp32 = x.to(torch.float32)

        # Output buffer (fp32 for compute)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Allocate per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=1024, num_warps=4
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        _icdf_ndtri_kernel[(1,)](
            torch.tensor(float(target_sparsity), dtype=torch.float32, device=x.device),
            B, S, F, icdf, BLOCK=1, num_warps=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=1024, num_warps=4
        )

        # Return in bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
