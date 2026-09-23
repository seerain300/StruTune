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
    start = (b * S + s) * F

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in BLOCK chunks
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        # Pointer arithmetic: X is [B, S, F] with F contiguous last dim
        ptrs = X + start + idx
        x = tl.load(ptrs, mask=mask, other=0.0)
        # Accumulate sums
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    var = sum_sq / F - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write out
    out_idx = pid
    tl.store(MEANS + out_idx, mean)
    tl.store(STDs + out_idx, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse standard normal CDF for scalar probability P (0 < P < 1).
    Uses Abramowitz and Stegun 5.2.23 (central region), with a5 negative.
    ICDF is a 1-element output tensor (float32).
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
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    result = poly * q / den

    tl.store(ICDF, result)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Sparsify X: OUT = max(0, X - (MEANS[row] + STDs[row] * ICDF)).
    MEANS: [B*S], STDs: [B*S], ICDF: scalar
    OUT: [B, S, F], X: [B, S, F] (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    start = (b * S + s) * F

    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    threshold = mean + std * tl.load(ICDF)

    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x_ptrs = X + start + idx
        y_ptrs = OUT + start + idx

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = x - threshold  # threshold is scalar
        y = tl.maximum(y, 0.0)
        tl.store(y_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward: computes adaptive sparsity threshold per (batch, seq)
        and applies ReLU(input - threshold) across the feature dimension.
        Maintains original dtype: returns bfloat16.
        """
        if target_sparsity == 0.0:
            return x  # no sparsity

        # Compute in fp32 for numeric stability
        x_fp32 = x.to(torch.float32).contiguous()
        B, S, F = x_fp32.shape

        # Allocate outputs and per-row stats
        out = torch.empty_like(x_fp32)  # fp32 working output
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Choose BLOCK and num_warps based on F
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

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
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
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
