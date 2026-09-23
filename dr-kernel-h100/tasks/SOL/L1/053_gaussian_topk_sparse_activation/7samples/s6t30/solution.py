import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_single_pass_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Single-pass kernel computing per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32)
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32), std = sqrt(sum(x^2) / F - mean^2)
    """
    pid = tl.program_id(axis=0)
    # Map program id to (b, s)
    b = pid // S
    s = pid % S

    # Compute base offset for the row
    row_offset = (b * S + s) * F

    # Accumulate sum and sum of squares across features in BLOCK chunks
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_offset + idx, mask=mask, other=0.0)
        # Reduce within the chunk
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        offs += BLOCK

    mean = sum_x / F
    # Population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_x2 / F - mean * mean
    # Ensure non-negative due to numerical rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results
    out_idx = b * S + s
    tl.store(MEANS + out_idx, mean)
    tl.store(STDs + out_idx, std)


@triton.jit
def _compute_sparsify_and_write_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Elementwise kernel: compute threshold per element and write sparsified output.
    OUT = ReLU(X - (MEANS[b*S + s] + STDs[b*S + s] * ICDF))
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_offset_x = (b * S + s) * F
    row_offset_out = (b * S + s) * F

    mean = tl.load(MEANS + (b * S + s))
    std = tl.load(STDs + (b * S + s))
    threshold = mean + std * ICDF  # ICDF is scalar broadcast

    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_offset_x + idx, mask=mask, other=0.0)
        y = tl.maximum(x - threshold, 0.0)
        tl.store(OUT + row_offset_out + idx, y, mask=mask)
        offs += BLOCK


@triton.jit
def _icdf_ndtri_scalar_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (standard normal) for a single scalar P in [0, 1].
    Uses Abramowitz & Stegun 5.2.23 central region approximation. Assumes P <= 0.5 path handled by caller.
    Stores result into ICDF (1-element tensor).
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5, r = q^2
    q = p - 0.5
    r = q * q

    # Coefficients (central region)
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

    # Evaluate numerator and denominator polynomials
    poly = (
        a6
        + r * (a5 + r * (a4 + r * (a3 + r * (a2 + r * a1))))
    )
    den = (
        1.0
        + r * (b5 + r * (b4 + r * (b3 + r * (b2 + r * b1))))
    )
    poly = poly * q
    icdf = poly / den

    # Store the result (ICDF is a 1-element tensor)
    tl.store(ICDF, icdf)


def _choose_block_and_warps(F: int):
    # Choose BLOCK as a power-of-two up to 2048 for robustness; mask handles tails.
    if F >= 4096:
        BLOCK = 2048
        num_warps = 8
    else:
        BLOCK = 1024
        num_warps = 4
    return BLOCK, num_warps


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-(batch, seq) mean and std across feature dimension.
        - Compute adaptive threshold using inverse normal CDF.
        - Apply ReLU(input - threshold).
        All numeric work is done by Triton kernels; host code only allocates outputs and launches kernels.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Compute in float32 for numerical stability
        x_fp32 = x.to(torch.float32)
        B, S, F = x_fp32.shape

        # Output buffer in float32
        out = torch.empty_like(x_fp32)

        # Allocate per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Choose BLOCK and num_warps for stats
        BLOCK_STATS, num_warps_stats = _choose_block_and_warps(F)
        # Launch single-pass rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_single_pass_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_scalar_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Choose BLOCK and num_warps for sparsify/write
        BLOCK_SP, num_warps_sp = _choose_block_and_warps(F)
        # Launch sparsify + write kernel
        _compute_sparsify_and_write_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
