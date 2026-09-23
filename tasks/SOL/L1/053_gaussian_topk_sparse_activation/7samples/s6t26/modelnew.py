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

    # Loop over features in chunks of BLOCK
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    Ff = tl.full((), F, tl.float32)
    mean = sum_val / Ff
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / Ff - mean * mean
    std = tl.sqrt(var)

    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a given scalar probability P (device scalar).
    Use Abramowitz & Stegun 5.2.23 central region approximation (valid for p ~ 0.5).
    Writes result to ICDF (device scalar).
    """
    # Load scalar probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    # Horner's method for numerator and denominator polynomials
    # Numerator: (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # ensure negative as per A&S
    a6 = 2.506628277459239e+00

    # Denominator: (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly_num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    poly_den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    icdf_val = poly_num / poly_den
    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, THRESH, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply elementwise sparsification: out = max(0, X - threshold)
    X: [B, S, F] input (float32)
    THRESH: [B*S] per-row thresholds (float32)
    OUT: [B, S, F] output (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F
    thr = tl.load(THRESH + pid)

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        y = tl.maximum(x - thr, 0.0)
        tl.store(OUT + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        - Compute per-row mean and std across feature dimension.
        - Adaptive cutoff = mean + std * icdf(target_sparsity).
        - Output = ReLU(x - cutoff).
        """
        # Ensure CUDA and float32 for compute
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape
        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Heuristic: larger BLOCK when safe to reduce loop iterations
        BLOCK_STATS = 2048 if F >= 2048 else 1024
        num_warps_stats = 8 if BLOCK_STATS == 2048 else 4

        # Rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Precompute per-row thresholds: mean + std * icdf
        threshold = means.view(B, S) + stds.view(B, S) * icdf
        threshold = threshold.view(B * S)

        # Sparsify + ReLU using precomputed thresholds
        BLOCK_SP = 2048 if F >= 2048 else 1024
        num_warps_sp = 8 if BLOCK_SP == 2048 else 4

        _sparsify_relu_kernel[grid](
            x_fp32, threshold, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)