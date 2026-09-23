import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes sum[row] and sumsq[row] to sum_out_ptr and sumsq_out_ptr.
    """
    row = tl.program_id(0)
    # Accumulators (float32)
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over feature dimension in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Each row is contiguous in memory: offset = row * F + cols
        offs = row * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    # Write results
    tl.store(sum_out_ptr + row, sum_val)
    tl.store(sumsq_out_ptr + row, sum_sq)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for a scalar p in [eps, 1-eps] using A&S 7.1.26 approximation.
    p_in_ptr: 1-element tensor holding input probability p.
    p_out_ptr: 1-element tensor to store result.
    """
    # Load input probability
    p = tl.load(p_in_ptr)
    # Clamp to avoid log(0)/log(1)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)
    # Constants for A&S approximation
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
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
    poly_b = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    z_mid = poly_mid * q_mid / poly_b

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1*q_up + c2)*q_up + c3)*q_up + c4)*q_up + c5)*q_up + c6) / \
           ((((d1*q_up + d2)*q_up + d3)*q_up + d4)*q_up + 1.0)

    # Select region
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_up = p > p_high

    z = tl.where(cond_low, z_low, 0.0) + tl.where(cond_mid, z_mid, 0.0) + tl.where(cond_up, z_up, 0.0)
    # Write result
    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    Apply activation: Y[row * F + cols] = max(0, X[row * F + cols] - (mean[row] + std[row] * multiplier))
    X_ptr: [ROWS*F], float32
    mean_ptr: [ROWS], float32
    std_ptr: [ROWS], float32
    multiplier_ptr: [1], float32
    Y_ptr: [ROWS*F], float32
    """
    row = tl.program_id(0)
    # Load scalar multiplier
    m = tl.load(multiplier_ptr)  # float32 scalar
    # Load mean and std for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    cutoff = mean + std * m

    # Iterate over features in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = row * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tuned parameters for performance
        self.block_size = 1024
        self.num_warps = 8
        self.num_stages = 2
        self.eps = 1e-7

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure dtype float32 for numerical stability
        x = inputs.to(torch.float32)
        B, S, F = x.shape
        rows = B * S
        # Flatten to [rows, F] contiguous
        X = x.contiguous().view(rows, F)

        # 1) Compute per-row sum and sum of squares in Triton
        sum_row = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(rows, dtype=torch.float32, device=x.device)
        grid = (rows,)
        row_stats_kernel[grid](
            X, sum_row, sumsq_row,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute mean and variance on host (unbiased=False semantics)
        mean = sum_row / F
        var = sumsq_row / F - mean * mean  # clamp negatives due to rounding if any
        # Ensure non-negative variance to avoid sqrt of small negative
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)  # small elementwise op on device

        # 3) Compute ndtri scalar in Triton
        p_in = x.new_tensor(target_sparsity).contiguous()  # 1-element tensor
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Apply activation via Triton
        Y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[grid](
            X, mean, std, std_multiplier, Y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        Y_out = Y.view(B, S, F).to(torch.bfloat16)
        return Y_out