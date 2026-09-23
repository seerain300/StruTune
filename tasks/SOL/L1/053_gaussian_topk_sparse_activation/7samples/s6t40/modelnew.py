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

    sum_val = 0.0
    sum_sq = 0.0

    # Loop over feature dimension in chunks of BLOCK
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Write outputs
    out_idx = b * S + s
    tl.store(MEANS + out_idx, mean)
    tl.store(STDs + out_idx, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for P using Abramowitz & Stegun 5.2.23 central region approximation.
    P: 1-element device tensor (float32) with probability p in (0, 1)
    ICDF: 1-element device tensor to store result
    """
    # Load probability p
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5

    # Coefficients for central region approximation
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

    # Polynomial evaluation
    r = q * q
    poly = (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    )
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Central region formula
    icdf_val = poly * q / denom
    # Store single element
    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply per-row adaptive threshold sparsification:
    out[b, s, f] = max(0, X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * ICDF))
    X: [B, S, F] input (float32)
    MEANS: [B*S] per-row means (float32)
    STDs: [B*S] per-row stds (float32)
    ICDF: 1-element tensor (float32) containing scalar z-score
    OUT: [B, S, F] output (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = b * S * F + s * F

    mean = tl.load(MEANS + (b * S + s))
    std = tl.load(STDs + (b * S + s))
    cutoff = mean + std * tl.load(ICDF)  # broadcast scalar

    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(OUT + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            return x

        # Compute in float32 for numerical stability
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Output as float32; will convert to bfloat16 at end
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Choose BLOCK and num_warps based on F for performance and robustness
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

        # Launch rowwise stats kernel (one program per (b, s) row)
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
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
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)