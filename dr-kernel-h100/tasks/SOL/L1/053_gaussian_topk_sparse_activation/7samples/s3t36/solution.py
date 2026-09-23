import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, var_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features,
    then mean = sum/F and variance = var = sumsq/F - mean^2.
    Writes mean[row] and var[row] to mean_out_ptr and var_out_ptr.
    """
    row = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over feature dimension in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Linearized index for row-major [ROWS, F]
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and variance for this row
    F_f = tl.float32(F)
    mean = sum_val / F_f
    var = (sum_sq / F_f) - mean * mean
    # Write results
    tl.store(mean_out_ptr + row, mean)
    tl.store(var_out_ptr + row, var)


@triton.jit
def sqrt_std_kernel(var_in_ptr, std_out_ptr, N: tl.int32):
    """
    Elementwise std = sqrt(var) over a 1D vector of length N.
    """
    idx = tl.program_id(0)
    if idx < N:
        var = tl.load(var_in_ptr + idx)
        std = tl.sqrt(var)  # Triton supports sqrt
        tl.store(std_out_ptr + idx, std)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) for a scalar p using
    Abramowitz & Stegun 7.1.26 approximation, writing result to p_out_ptr[0].
    Clamp p to [eps, 1-eps].
    """
    # Load p
    p = tl.load(p_in_ptr)
    # Clamp
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    result = tl.where(p < p_low, result_low, 0.0)

    # Central region
    q = p - 0.5
    r = q * q
    result_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    result = tl.where((p >= p_low) & (p <= p_high), result_mid, result)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    result = tl.where(p > p_high, result_up, result)

    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    For each row r (0..ROWS-1), compute cutoff = mean[r] + std[r] * multiplier,
    then apply y = max(0, X[r, :] - cutoff) and store to Y.
    """
    row = tl.program_id(0)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    mult = tl.load(multiplier_ptr)  # scalar multiplier
    cutoff = mean + std * mult
    base = row * F
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = base + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        # Tunable parameters
        self.block_size = 1024
        self.num_warps = 8
        self.num_stages = 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes per-row mean and std across feature dimension, then
        y = max(0, x - (mean + std * ndtri(target_sparsity))).
        Returns output cast to bfloat16, same shape as input.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Flatten to [rows, F] where rows = B * S
        B, S, F = x.shape
        rows = B * S

        # Work in float32 for numerical stability
        x32 = x.to(torch.float32).contiguous()
        x_flat = x32.view(rows * F)

        # 1) Compute per-row sum and sumsq using Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        # Launch reduction kernel
        grid_rows = (rows,)
        row_stats_kernel[grid_rows](
            x_flat, mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute variance per row in Triton
        var = torch.empty(rows, dtype=torch.float32, device=x32.device)
        grid_var = (rows,)
        sqrt_std_kernel[grid_var](
            sumsq, var, N=rows,
            num_warps=4,  # small elementwise op
            num_stages=1,
        )

        # 3) Compute scalar ndtri(target_sparsity) in Triton
        p_in = x32.new_tensor(target_sparsity).contiguous()  # 1-element tensor
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, var.sqrt(), std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
