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

    # Compute base pointer offset for the row: b*S + s
    base = b * S + s
    # Accumulators for sum and sum of squares
    total = 0.0
    total2 = 0.0

    # Iterate over feature dimension in BLOCK chunks
    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        # Row pointer: X is [B, S, F] contiguous in last dim, so stride along F is 1
        x_ptrs = X + base * F + offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)

    n = F
    mean = total / n
    var = total2 / n - mean * mean
    # Population std (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    MEANS[base] = mean
    STDs[base] = std


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Compute output = ReLU(X - (MEANS[b*s] + STDs[b*s] * ICDF)) for each element.
    X, OUT: [B, S, F] float32
    MEANS, STDs: [B*S] float32
    ICDF: scalar float32
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S + s
    mean = MEANS[base]
    std = STDs[base]
    threshold = mean + std * ICDF

    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F

        x_ptrs = X + base * F + offsets
        out_ptrs = OUT + base * F + offsets

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _icdf_ndtri_scalar_kernel(P, ICDF):
    """
    Compute inverse normal CDF for a scalar probability P (0 < P < 1) using
    Abramowitz & Stegun 5.2.23 central region approximation.
    Stores the result in ICDF (float32 scalar on device).
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as per A&S
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


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized sparsification:
        - Compute per-row (batch, seq) mean and std across the last dimension (features).
        - Compute icdf(target_sparsity) using Triton scalar kernel.
        - Apply ReLU(x - (mean + std * icdf)) and return in bfloat16.
        """
        # If no sparsity requested, return as is
        if target_sparsity == 0.0:
            return x

        # Ensure input is contiguous float32
        x = x.contiguous()
        B, S, F = x.shape
        x_fp32 = x.to(torch.float32)

        # Allocate outputs for per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Choose BLOCK and num_warps for robustness and performance
        BLOCK = 1024 if F < 2048 else 2048
        num_warps = 4 if BLOCK == 1024 else 8

        # Launch rowwise stats kernel: one program per (b, s)
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=num_warps, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton scalar kernel
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_scalar_kernel[(1,)](
            p_dev, icdf, num_warps=1, num_stages=1
        )

        # Output buffer
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Launch sparsify + ReLU kernel: one program per (b, s), iterate over features
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=num_warps, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
