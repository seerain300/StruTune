import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F, sumsq[row] to mean_out_ptr and sumsq_out_ptr.
    """
    row_id = tl.program_id(axis=0)
    # Accumulate sum and sum of squares in float32.
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x_row = tl.load(X_ptr + row_id * F + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x_row, axis=0)
        sumsq_val += tl.sum(x_row * x_row, axis=0)

    mean = sum_val / F
    sumsq = sumsq_val
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq)


@triton.jit
def var_std_kernel(sumsq_ptr, mean_ptr, std_ptr, F: tl.int32):
    """
    Elementwise compute std = sqrt(sumsq/F - mean^2) for each row.
    Assumes input tensors are 1D with length ROWS.
    """
    row_id = tl.program_id(axis=0)
    var = tl.load(sumsq_ptr + row_id) / F - tl.load(mean_ptr + row_id) * tl.load(mean_ptr + row_id)
    # Guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(std_ptr + row_id, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) for scalar p in [eps, 1-eps]
    using Abramowitz & Stegun 7.1.26 approximation.
    Writes the result into p_out_ptr (single-element tensor).
    """
    p = tl.load(p_in_ptr)  # float32 scalar
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants for approximation
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
    mask_low = p < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    # Polynomial in q for low region
    poly_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    denom_low = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    res_low = poly_low / denom_low

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    res_mid = poly_mid * q_mid / denom_mid

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    res_high = -poly_high / denom_high

    # Select result based on region
    res = tl.where(mask_low, res_low, 0.0)
    res = tl.where(mask_mid, res_mid, res)
    res = tl.where(mask_high, res_high, res)

    tl.store(p_out_ptr, res)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Out_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    Elementwise apply y = max(0, x - (mean + std * multiplier)) for each row.
    Assumes X_ptr, mean_ptr, std_ptr are 2D with shape [ROWS, F], Out_ptr is [ROWS*F].
    multiplier_ptr is 1-element tensor with scalar multiplier.
    """
    row_id = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    mult = tl.load(multiplier_ptr)  # scalar

    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x_row = tl.load(X_ptr + row_id * F + cols, mask=mask, other=0.0)
        cutoff = mean + std * mult
        y_row = tl.maximum(x_row - cutoff, 0.0)
        tl.store(Out_ptr + row_id * F + cols, y_row, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size=2048, num_warps=8, num_stages=2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = 1e-7

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the Gaussian-based top-k sparse activation.
        Computes adaptive sparsity threshold per row based on mean and std,
        then applies y = max(0, x - (mean + std * ndtri(target_sparsity))).
        Returns tensor of same shape as input, cast to bfloat16.
        """
        # Flatten [B, S, F] to [rows, F]
        B, S, F = x.shape
        rows = B * S
        x_flat = x.contiguous().view(rows * F)
        # Cast to float32 for statistics and activation
        x32 = x_flat.view(rows, F).to(torch.float32)

        # Allocate outputs for stats
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)

        # Launch row stats kernel
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x32, mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Compute ndtri scalar for target_sparsity in Triton
        p_in = x32.new_tensor(target_sparsity)  # 1-element device tensor
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # Compute std per row in Triton
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        var_std_kernel[grid_stats](
            sumsq, mean, std,
            ROWS=rows, F=F,
            num_warps=4,
            num_stages=2,
        )

        # Apply activation in Triton
        out = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x32, mean, std, std_multiplier, out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Reshape and cast to bfloat16 to match original output
        y_out = out.view(B, S, F).to(torch.bfloat16)
        return y_out