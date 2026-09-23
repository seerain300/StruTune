import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), contiguous
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base linear offset for this row
    base = b * S + s
    # Compute row base pointer offset in 1D contiguous layout: row_offset = (b*S + s) * F
    row_offset = base * F

    # Accumulators
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over features in chunks of BLOCK
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        # Linear pointer to elements of this row
        ptr = row_offset + idx
        x = tl.load(X + ptr, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = F
    mean = sum_x / n
    # population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_x2 / n - mean * mean
    std = tl.sqrt(var)

    # Write outputs
    tl.store(MEANS + base, mean)
    tl.store(STDs + base, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    For each (b, s) row, compute threshold = mean + std * ICDF, then write
    OUT[b, s, f] = max(0, X[b, s, f] - threshold).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S + s
    row_offset_x = base * F
    row_offset_out = base * F

    mean = tl.load(MEANS + base)
    std = tl.load(STDs + base)
    threshold = mean + std * ICDF

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_offset_x + idx, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(OUT + row_offset_out + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_scalar_kernel(P, ICDF):
    """
    Compute inverse normal CDF (quantile) for p in (0, 1) using Abramowitz & Stegun 5.2.23 central region.
    Stores result into ICDF (1-element tensor, float32).
    """
    # Load p
    p = tl.load(P)
    # Central region approximation: q = p - 0.5
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
    icdf_val = poly * q / den
    tl.store(ICDF, icdf_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized sparsity activation.
        - Compute per-row mean and population std across last dim (features).
        - threshold = mean + std * icdf(target_sparsity)
        - output = ReLU(x - threshold), returned in bfloat16.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous and compute in fp32
        x = x.contiguous()
        x_fp32 = x.to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Fixed BLOCK and warps for robustness
        BLOCK = 1024
        num_warps_stats = 4
        num_warps_sp = 4

        # Launch rowwise stats kernel: one program per (b, s)
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton scalar kernel
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_scalar_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Output buffer
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Launch sparsify + ReLU kernel: one program per (b, s), iterate over features
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)