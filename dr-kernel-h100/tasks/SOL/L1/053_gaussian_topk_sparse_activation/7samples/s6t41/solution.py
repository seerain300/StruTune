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

    # Compute base pointer for this row
    row_ptr = X + b * S * F + s * F

    total = 0.0
    total2 = 0.0
    # Loop over features in BLOCK chunks
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)

    n = F
    mean = total / n
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = total2 / n - mean * mean
    std = tl.sqrt(var)

    # Write results
    out_index = pid
    tl.store(MEANS + out_index, mean)
    tl.store(STDs + out_index, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse-normal CDF for scalar probability P[0] using Abramowitz & Stegun 5.2.23 central region formula.
    ICDF[0] = z such that erf(z/sqrt(2)) ≈ P, approximated via rational polynomial.
    We only use the central region formula for p in [0.5, 1.0].
    a5 is negative: -3.066479806614716e+01 to ensure correctness.
    """
    p = tl.load(P)
    # q = p - 0.5
    q = p - 0.5
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative
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
    Apply sparsification: OUT[b, s, f] = max(0, X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * ICDF))
    where ICDF is a scalar (same for all rows).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_ptr = X + b * S * F + s * F
    out_ptr = OUT + b * S * F + s * F

    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    cutoff = mean + std * tl.load(ICDF)
    # Apply ReLU with adaptive threshold: max(0, x - cutoff)
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Returns sparsified tensor of same shape as input, in bfloat16.
        """
        if target_sparsity == 0.0:
            # No sparsity requested, just return input cast to bfloat16
            return x.to(torch.bfloat16)

        # Ensure fp32 for compute
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        out = torch.empty_like(x_fp32)

        # Choose BLOCK based on F for better throughput
        BLOCK_STATS = 2048 if F >= 2048 else 1024
        num_warps_stats = 8 if BLOCK_STATS == 2048 else 4
        num_stages_stats = 2

        # Launch rowwise stats kernel (one program per (b, s) row)
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        BLOCK_SP = 2048 if F >= 2048 else 1024
        num_warps_sp = 8 if BLOCK_SP == 2048 else 4
        num_stages_sp = 2

        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
