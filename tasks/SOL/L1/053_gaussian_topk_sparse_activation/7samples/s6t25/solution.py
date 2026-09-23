import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), row-major
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Pointer to the start of row (b, s, :)
    row_start = (b * S + s) * F  # one program per row, contiguous along F
    offsets = tl.arange(0, BLOCK)
    idx = row_start + offsets
    mask = offsets < F

    # Accumulators (scalars)
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in BLOCK chunks
    for start in range(0, F, BLOCK):
        idx = start + offsets
        mask = idx < F
        x = tl.load(X + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and population std (unbiased=False)
    F_f = tl.full((), F, tl.float32)
    mean = sum_val / F_f
    var = sum_sq / F_f - mean * mean
    # std = sqrt(max(var, 0)) for numerical safety
    std = tl.sqrt(tl.maximum(var, 0.0))

    # Write outputs: linear index for [B*S]
    out_idx = b * S + s
    tl.store(MEANS + out_idx, mean)
    tl.store(STDs + out_idx, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for scalar P in (0,1) using Abramowitz & Stegun 5.2.23 (central region).
    P: 1-element tensor (float32)
    ICDF: 1-element tensor (float32)
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    # Coefficients (float64 literals; Triton will handle in f32, acceptable here)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # note: negative as per A&S
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Compute polynomial
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    num = poly * q  # central region uses q = p - 0.5

    icdf_val = num / den
    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = max(0, X - (MEANS + STDs * ICDF)).
    One program per row (b, s), iterate over F in BLOCK chunks.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F

    mean = tl.load(MEANS + (b * S + s))
    std = tl.load(STDs + (b * S + s))
    cutoff = mean + std * tl.load(ICDF)  # ICDF is scalar, broadcast

    offsets = tl.arange(0, BLOCK)
    for start in range(0, F, BLOCK):
        idx = start + offsets
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)  # ReLU(x - cutoff)
        tl.store(OUT + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Computes per-(B, S) mean and std across last dim.
        - Computes icdf(target_sparsity) with A&S 5.2.23 central region.
        - Applies ReLU(x - (mean + std * icdf)) per element.
        """
        if target_sparsity == 0.0:
            return x

        # Ensure float32 compute, keep original bfloat16 behavior in return
        x = x.to(torch.float32).contiguous()
        B, S, F = x.shape

        # Allocate outputs for stats and final
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        out = torch.empty_like(x, dtype=torch.float32)

        # Select BLOCK and num_warps based on F (constexpr for Triton)
        if F >= 4096:
            BLOCK_STATS = BLOCK_SP = 4096
            num_warps_stats = 8
            num_warps_sp = 8
            num_stages = 3
        elif F >= 2048:
            BLOCK_STATS = BLOCK_SP = 2048
            num_warps_stats = 8
            num_warps_sp = 8
            num_stages = 3
        else:
            BLOCK_STATS = BLOCK_SP = 1024
            num_warps_stats = 4
            num_warps_sp = 4
            num_stages = 2

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](x, B, S, F, means, stds, BLOCK=BLOCK_STATS,
                                    num_warps=num_warps_stats, num_stages=num_stages)

        # Compute icdf for target_sparsity (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1)

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](x, means, stds, icdf, out, B, S, F,
                                    BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages)

        # Return in bfloat16 to match original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
