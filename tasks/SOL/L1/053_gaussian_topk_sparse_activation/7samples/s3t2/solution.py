import math
import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(
    X_ptr,            # *float32, input tensor, shape [ROWS, F]
    MEAN_ptr,         # *float32, output per-row mean, shape [ROWS]
    VAR_ptr,          # *float32, output per-row var (unbiased=False), shape [ROWS]
    ROWS,             # int
    F,                # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    # Bounds check
    if row_id >= ROWS:
        return

    # Accumulate sum and sum of squares for this row
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Compute linear index for this row and columns
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Reduce within the tile
        # Note: mask ensures out-of-range loads are 0.0
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # unbiased=False
    # Store results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(VAR_ptr + row_id, var)


@triton.jit
def _ndtri_kernel(
    P_ptr,            # *float32, input scalar p (0 < p < 1), shape [1]
    OUT_ptr,          # *float32, output scalar z, shape [1]
):
    # Single program computes ndtri for the scalar
    # Constants (Abramowitz & Stegun 7.1.26)
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

    # Load p
    p = tl.load(P_ptr)  # scalar

    # Masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Initialize result
    result = 0.0

    # Compute ndtri piecewise
    if mask_low:
        # Lower region: use transformed variable sqrt(-2*log(p))
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        result = poly / denom
    elif mask_mid:
        # Central region: p - 0.5
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        result = poly * q / denom
    else:
        # Upper region: use transformed variable sqrt(-2*log(1-p))
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        result = poly / denom

    tl.store(OUT_ptr, result)


@triton.jit
def _relu_threshold_kernel(
    X_ptr,            # *float32, input tensor, shape [ROWS, F]
    MEAN_ptr,         # *float32, per-row mean, shape [ROWS]
    VAR_ptr,          # *float32, per-row var, shape [ROWS]
    MULTIPLIER_ptr,   # *float32, scalar ndtri(target_sparsity), shape [1]
    Y_ptr,            # *float32, output tensor, shape [ROWS, F]
    ROWS: tl.constexpr,  # not used directly, kept for signature symmetry
    F,                # int
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return

    # Load per-row mean and std
    mean = tl.load(MEAN_ptr + row_id)
    var = tl.load(VAR_ptr + row_id)
    std = tl.sqrt(var)
    multiplier = tl.load(MULTIPLIER_ptr)  # scalar
    cutoff = mean + std * multiplier

    # Iterate over columns and apply activation
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff  # broadcast cutoff scalar
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size=256, num_warps=8, num_stages=4):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run:
        - Computes per-row mean and std in Triton.
        - Computes inverse normal CDF for target_sparsity in Triton (scalar).
        - Applies ReLU(input - (mean + std * multiplier)) in Triton.
        - Returns output cast to bfloat16.
        """
        # Ensure dtype float32 for stable stats; original code used float32 internally anyway
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        rows = B * S

        # Allocate outputs for stats and final activation
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        var = torch.empty(rows, dtype=torch.float32, device=x.device)
        y = torch.empty_like(x, dtype=torch.float32)  # activation output (float32)

        # Launch Triton kernel to compute row-wise mean and var
        grid_stats = (rows,)
        _row_stats_kernel[grid_stats](
            x, mean, var,
            rows, F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Compute std multiplier using Triton kernel (scalar evaluation of ndtri)
        p_tensor = torch.tensor([target_sparsity], dtype=torch.float32, device=x.device)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_kernel[(1,)](
            p_tensor, std_multiplier,
            num_warps=1,
            num_stages=1,
        )

        # Apply ReLU threshold per element in Triton
        grid_act = (rows,)
        _relu_threshold_kernel[grid_act](
            x, mean, var, std_multiplier, y,
            rows, F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Cast to bfloat16 to match original
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
