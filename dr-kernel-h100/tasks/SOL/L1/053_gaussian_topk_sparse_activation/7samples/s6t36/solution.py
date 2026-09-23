import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr, NUM_ITERS: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32)
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer for this row
    base = b * S + s
    row_ptr = X + base * F

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over tiles of size BLOCK
    for i in range(NUM_ITERS):
        start = i * BLOCK
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    n = F  # population std (unbiased=False), since we sum all elements
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # Avoid tiny negative due to roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    MEANS[base] = mean
    STDs[base] = std


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a single probability P (scalar tensor) using
    Abramowitz and Stegun 5.2.23 central region approximation. Returns scalar ICDF.
    """
    # Load probability
    p = tl.load(P)  # scalar float
    # Central region: q = p - 0.5, r = q^2
    q = p - 0.5
    r = q * q
    # Polynomial with coefficients
    # a1..a6 and b1..b5 are constants; a5 must be negative.
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
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    inv = poly / den
    icdf = q * inv
    tl.store(ICDF, icdf)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr, NUM_ITERS: tl.constexpr):
    """
    Apply sparsification: OUT = ReLU(X - (MEANS[pid] + STDs[pid] * ICDF)).
    One program per (b, s) row. Iterate over features in tiles of BLOCK.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S + s
    mean = MEANS[base]
    std = STDs[base]
    thresh = mean + std * tl.load(ICDF)

    row_ptr = X + base * F
    out_ptr = OUT + base * F

    for i in range(NUM_ITERS):
        start = i * BLOCK
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-(batch, seq) mean and std across last dim.
        - Compute icdf for target_sparsity in Triton (Abramowitz & Stegun 5.2.23 central region).
        - Apply sparsification: ReLU(x - (mean + std * icdf)), returning bfloat16.
        """
        if target_sparsity == 0.0:
            return x  # no sparsity

        # Compute in fp32 for numeric stability
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Output tensor
        out = torch.empty_like(x_fp32)

        # Choose BLOCK and num_warps heuristically
        if F >= 2048:
            BLOCK_STATS = BLOCK_SP = 2048
            num_warps_stats = 8
            num_warps_sp = 8
        else:
            BLOCK_STATS = BLOCK_SP = 1024
            num_warps_stats = 4
            num_warps_sp = 4

        # Number of iterations for loops (constexpr meta-parameters)
        NUM_ITERS_STATS = (F + BLOCK_STATS - 1) // BLOCK_STATS
        NUM_ITERS_SP = (F + BLOCK_SP - 1) // BLOCK_SP

        # Launch rowwise stats kernel (one program per row)
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds,
            BLOCK=BLOCK_STATS, NUM_ITERS=NUM_ITERS_STATS,
            num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](p_dev, icdf, num_warps=1, num_stages=1)

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F,
            BLOCK=BLOCK_SP, NUM_ITERS=NUM_ITERS_SP,
            num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
