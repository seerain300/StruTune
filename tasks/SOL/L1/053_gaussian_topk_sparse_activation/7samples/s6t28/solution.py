import torch
import triton
import triton.language as tl


@triton.jit
def _compute_mean_std_and_sparsify_kernel(
    X, OUT,
    B, S, F,
    MEANS, STDs,
    ICDF,
    BLOCK: tl.constexpr
):
    """
    Single-kernel pass:
    - For each row (b, s), reduce across features to compute mean and std (population).
    - Compute threshold = mean + std * icdf(target_sparsity).
    - Compute output = max(0, x - threshold) for all features.
    All tensors are [B, S, F].
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointers for this row
    base_x = X + b * S * F + s * F
    base_out = OUT + b * S * F + s * F

    # Vector of feature offsets
    offs = tl.arange(0, BLOCK)

    # Accumulate sum and sum of squares for mean and std
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over feature dimension in BLOCK chunks
    for start in range(0, F, BLOCK):
        idx = start + offs
        mask = idx < F
        x = tl.load(base_x + idx, mask=mask, other=0.0)
        # accumulate
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    # Compute mean and std for this row (population std, unbiased=False)
    F_f = tl.float32(F)
    mean = sum_x / F_f
    var = sum_x2 / F_f - mean * mean
    # Ensure non-negative variance due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store per-row statistics (cast to float32)
    row_idx = b * S + s
    tl.store(MEANS + row_idx, mean)
    tl.store(STDs + row_idx, std)

    # Compute threshold and apply sparsification
    threshold = mean + std * ICDF

    # Second pass: write output = max(0, x - threshold)
    for start in range(0, F, BLOCK):
        idx = start + offs
        mask = idx < F
        x = tl.load(base_x + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(base_out + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF):
    """
    Compute inverse normal CDF for a single scalar probability P (device tensor of shape [1])
    using Abramowitz & Stegun 5.2.23 central region approximation.
    Writes result into ICDF (device tensor of shape [1]).
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5

    # Coefficients
    # a1..a5 and b1..b5 (population std formula 26.2.23, central region)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # must be negative

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly / den  # sign is correct in central region

    tl.store(ICDF, icdf)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run:
        - Compute mean and std per (batch, seq) row.
        - Compute adaptive threshold using inverse normal CDF of target_sparsity.
        - Apply ReLU(x - threshold) across features.
        Returns output in bfloat16, matching original behavior.
        """
        if target_sparsity == 0.0:
            return x

        # Compute in float32 for numeric stability
        device = x.device
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=device)
        out = torch.empty((B, S, F), dtype=torch.float32, device=device)

        # Choose a robust BLOCK size (constexpr). Avoid very large to prevent codegen issues.
        BLOCK = 1024
        num_warps = 4
        num_stages = 2

        # Launch single Triton kernel over rows
        grid = (B * S,)
        _compute_mean_std_and_sparsify_kernel[grid](
            x_fp32, out,
            B, S, F,
            means, stds,
            ICDF=torch.tensor(0.0, dtype=torch.float32, device=device),
            BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        p_dev = torch.empty((), dtype=torch.float32, device=device)
        p_dev.fill_(float(target_sparsity))
        icdf_dev = torch.empty((), dtype=torch.float32, device=device)
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf_dev, num_warps=1, num_stages=1
        )

        # Cast back to bfloat16 to match original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
