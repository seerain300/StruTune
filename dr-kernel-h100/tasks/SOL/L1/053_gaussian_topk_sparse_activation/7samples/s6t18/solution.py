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

    # Base pointer to this row
    row_base = X + b * S * F + s * F

    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in chunks of BLOCK
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_base + idx, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    var = sum_sq / F - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for given probability P (device scalar).
    Uses Abramowitz & Stegun 5.2.23 central region approximation.
    Writes scalar result to ICDF.
    """
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q
    # Polynomial coefficients
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

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf_val = poly / den
    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsity: OUT = max(0, X - (MEANS + STDs * ICDF))
    MEANS and STDs are per-row scalars; broadcast across features.
    """
    pid = tl.program_id(axis=0)
    # Load per-row mean and std
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    threshold = mean + std * ICDF

    # Base pointers
    row_in = X + pid * F
    row_out = OUT + pid * F

    # Loop over features in chunks of BLOCK
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_in + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(row_out + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        1) Compute per-row mean and std over last dim (features)
        2) Compute icdf(target_sparsity) using Triton
        3) Apply sparsity: max(0, x - (mean + std * icdf))
        Return in bfloat16, matching original behavior.
        """
        # Ensure float32 compute
        x_fp32 = x.to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Fixed, robust configuration
        BLOCK = 1024
        num_warps_stats = 4
        num_stages_stats = 2

        # Launch rowwise stats kernel: one program per (batch, seq) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Allocate output
        out = torch.empty_like(x_fp32)

        # Launch sparsify + ReLU kernel with the same BLOCK/warps/stages
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
