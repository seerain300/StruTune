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

    # Row base index in flattened [B*S, F]
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
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results for this row
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a single scalar probability using A&S 5.2.23.
    P: 1-element device tensor with probability (float32)
    OUT: 1-element device tensor for result (float32)
    """
    p = tl.load(P)  # load scalar probability
    # Clamp to safe range
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

    # Constants for A&S 5.2.23
    p_low = 0.02425
    p_high = 1.0 - p_low

    # a coefficients
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    # b coefficients
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # c coefficients
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    # d coefficients
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Masks for regions
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Compute q for each region
    # Lower region: q = sqrt(-2*log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Mid region: q = p - 0.5
    q_mid = p - 0.5
    y_mid = (((((a1 * q_mid * q_mid + a2) * q_mid + a3) * q_mid + a4) * q_mid + a5) * q_mid + a6) * q_mid / \
            (((((b1 * q_mid * q_mid + b2) * q_mid + b3) * q_mid + b4) * q_mid + b5) * q_mid + 1.0)

    # Upper region: q = sqrt(-2*log(1-p))
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select region results
    y = tl.where(mask_low, y_low, 0.0)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_high, y_up, y)

    # Store icdf
    tl.store(OUT, y)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Elementwise sparsification with ReLU threshold:
    threshold = MEANS[b*s] + STDs[b*s] * ICDF
    OUT = max(0, X - threshold)
    One program per (b, s) row.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    thr_mult = tl.load(ICDF)  # scalar
    threshold = mean + std * thr_mult

    base = (b * S + s) * F

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(OUT + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version of run:
        - Compute per-(batch, seq) mean and population std (unbiased=False).
        - Compute inverse normal CDF for target_sparsity via A&S approximation in Triton.
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

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=1024, num_warps=4
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        # Pass probability via 1-element device tensor
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=1024, num_warps=4
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)