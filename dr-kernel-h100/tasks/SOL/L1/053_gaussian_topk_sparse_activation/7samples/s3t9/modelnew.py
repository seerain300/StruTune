import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(x_ptr, mean_ptr, var_ptr, F: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute per-row mean and variance along the feature dimension F.
    x_ptr points to a [rows, F] flattened tensor.
    mean_ptr[row] = mean of row, var_ptr[row] = variance of row (unbiased=False).
    """
    row = tl.program_id(0)
    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in tiles
    offs = tl.arange(0, BLOCK)
    for start in range(0, F, BLOCK):
        idx = start + offs  # [BLOCK]
        mask = idx < F
        # Compute addresses: for row and idx, flattened index = row * F + idx
        ptrs = x_ptr + row * F + idx
        x = tl.load(ptrs, mask=mask, other=0.0)
        # Reduce tile into scalars
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # unbiased=False
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)


@triton.jit
def _ndtri_scalar_kernel(p_ptr, out_ptr):
    """
    Compute inverse standard normal CDF (quantile) for a single p.
    p_ptr points to a 1-element float32 tensor with the target sparsity.
    out_ptr points to a 1-element float32 tensor to write the result.
    Uses Abramowitz and Stegun 7.1.26 piecewise approximation.
    """
    p = tl.load(p_ptr)
    # Clamp p to (eps, 1-eps) to avoid log(0) and NaNs
    eps = 1e-7
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants
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

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Piecewise approximation
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    lower = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q2 = p - 0.5
    r2 = q2 * q2
    central = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * q2 / \
              (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    upper = -(((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6) / \
            ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)

    # Select region
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = ~mask_low & ~mask_mid  # p > p_high

    # Compute result
    result = tl.zeros_like(p)
    result = tl.where(mask_low, lower, result)
    result = tl.where(mask_mid, central, result)
    result = tl.where(mask_high, upper, result)

    tl.store(out_ptr, result)


@triton.jit
def _apply_activation(x_ptr, mean_ptr, var_ptr, multiplier_ptr, out_ptr,
                       rows: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    """
    Apply elementwise activation: y = max(0, x - (mean + std * multiplier))
    x_ptr points to [rows, F] flattened tensor.
    mean_ptr[row], var_ptr[row] are per-row statistics (float32).
    multiplier_ptr[0] is scalar std multiplier (float32).
    out_ptr writes float32 results.
    """
    row = tl.program_id(0)
    # Compute std from var in-kernel
    std = tl.sqrt(tl.load(var_ptr + row))
    cutoff = tl.load(mean_ptr + row) + std * tl.load(multiplier_ptr)
    # Iterate over features and apply activation
    offs = tl.arange(0, BLOCK)
    for start in range(0, F, BLOCK):
        idx = start + offs
        mask = idx < F
        x = tl.load(x_ptr + row * F + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row * F + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 4):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure dtype is float32 for compute; original code uses float32 for stats and activation.
        x_f32 = x.to(torch.float32)
        B, S, F = x_f32.shape
        rows = B * S

        # 1) Compute per-row mean and var using Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x_f32.device)
        var = torch.empty(rows, dtype=torch.float32, device=x_f32.device)

        _row_stats_kernel[(rows,)](
            x_f32.view(rows, F),
            mean, var,
            F=F, BLOCK=self.block_size,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # 2) Compute std_multiplier = ndtri(target_sparsity) via Triton
        # Create a 1-element device tensor holding the scalar sparsity; Triton kernel reads it.
        p = torch.empty(1, dtype=torch.float32, device=x_f32.device)
        p[0] = float(target_sparsity)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        _ndtri_scalar_kernel[(1,)](
            p, std_multiplier,
            num_warps=1, num_stages=1
        )

        # 3) Apply activation in Triton, writing float32 output
        y = torch.empty_like(x_f32, shape=(rows, F), dtype=torch.float32, device=x_f32.device)

        _apply_activation[(rows,)](
            x_f32.view(rows, F), mean, var, std_multiplier, y,
            rows=rows, F=F, BLOCK=self.block_size,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

        # 4) Reshape back to [B, S, F] and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out