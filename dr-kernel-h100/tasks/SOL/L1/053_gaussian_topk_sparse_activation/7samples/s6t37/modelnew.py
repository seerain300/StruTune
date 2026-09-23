import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, MEANS, STDs, B: tl.constexpr, S: tl.constexpr, F: tl.constexpr, BLOCK: tl.int32):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32, contiguous)
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F  # element index in flattened X

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    offset = 0
    while offset < F:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        # Flattened pointer for this row
        x_ptrs = X + row_start + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        # Reduce within the tile
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        offset += BLOCK

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # population std (unbiased=False)
    std = tl.sqrt(var)
    # Store results for this row
    MEANS[pid] = mean
    STDs[pid] = std


@triton.jit
def _icdf_ndtri_kernel(P, ICDF):
    """
    Compute inverse normal CDF (quantile) for probability P (scalar).
    Uses Abramowitz & Stegun 5.2.23 central region approximation.
    P: [1] device tensor of float32
    ICDF: [1] device tensor of float32
    """
    p = tl.load(P)  # scalar
    # Central region: q = p - 0.5, r = q^2
    q = p - 0.5
    r = q * q

    # Coefficients (central region)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative, important
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    num = poly * q
    icdf_val = num / den

    ICDF[0] = icdf_val


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B: tl.constexpr, S: tl.constexpr, F: tl.constexpr, BLOCK: tl.int32):
    """
    Sparsify each element: OUT[b, s, f] = max(0, X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * ICDF[0])).
    One program per row (b, s). Iterate over features in chunks of BLOCK.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F

    mean = MEANS[pid]
    std = STDs[pid]
    threshold = mean + std * ICDF[0]

    offset = 0
    while offset < F:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        x_ptrs = X + row_start + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        sparse = x - threshold
        sparse = tl.maximum(sparse, 0.0)  # ReLU
        out_ptrs = OUT + row_start + idx
        tl.store(out_ptrs, sparse, mask=mask)
        offset += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes per-(batch, seq) row mean and std, then sparsifies:
          threshold = mean + std * ndtri(target_sparsity)
          out = max(0, x - threshold)
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            # Keep dtype behavior consistent: return original dtype (here input is bfloat16)
            return x

        # Ensure contiguous and compute in fp32
        x_fp32 = x.contiguous().to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate outputs and stats buffers
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        out = torch.empty_like(x_fp32)

        # Launch rowwise stats kernel: one program per (b, s) row
        BLOCK_STATS = 1024
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, means, stds, B, S, F, BLOCK_STATS,
            num_warps=4, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        BLOCK_SP = 1024
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK_SP,
            num_warps=4, num_stages=2
        )

        # Return in original dtype (bfloat16), matching original behavior
        return out.to(torch.bfloat16)