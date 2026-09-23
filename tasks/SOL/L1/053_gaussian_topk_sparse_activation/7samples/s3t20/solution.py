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
    # Each program handles one row; we loop across the feature dimension in tiles.
    # Accumulate sum and sum of squares in float32.
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Compute linear indices for the row
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / F
    # Store mean and sumsq (we'll compute std on host: std = sqrt(sumsq/F - mean^2) )
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq_val)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for a single probability p_in_ptr[0]
    using Abramowitz and Stegun 7.1.26 approximation and write to p_out_ptr[0].
    Clamp p to [eps, 1-eps] to avoid log(0)/log(1) edge cases.
    """
    p = tl.load(p_in_ptr).to(tl.float32)
    # Clamp
    p = tl.maximum(eps, p)
    p = tl.minimum(1.0 - eps, p)

    # Constants for A&S 7.1.26 approximation
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
    # We'll compute piecewise in registers and write once at the end
    result_low = tl.zeros((), dtype=tl.float32)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                 ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    result_mid = tl.zeros((), dtype=tl.float32)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    result_mid = poly_mid / den_mid

    # Upper region
    mask_high = p > p_high
    result_high = tl.zeros((), dtype=tl.float32)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                  ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select based on masks; Triton supports per-branch computation via masks
    result = tl.zeros((), dtype=tl.float32)
    result = tl.where(mask_low, result_low, result)
    result = tl.where(mask_mid, result_mid, result)
    result = tl.where(mask_high, result_high, result)

    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), load mean and std, compute cutoff = mean + std * multiplier,
    then write Y[row, :] = max(0, X[row, :] - cutoff).
    """
    row_id = tl.program_id(axis=0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id).to(tl.float32)
    std = tl.load(std_ptr + row_id).to(tl.float32)
    mult = tl.load(multiplier_ptr).to(tl.float32)  # scalar multiplier

    cutoff = mean + std * mult

    # Iterate over features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - cutoff
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 2048, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Flatten to [rows, F], keep dtype as float32 for numerics
        x = inputs
        B, S, F = x.shape
        rows = B * S
        x32 = x.to(torch.float32).contiguous()

        # 1) Compute per-row sum and sumsq in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32.view(rows, F),
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std_multiplier = inverse standard normal CDF(target_sparsity) in Triton
        p_in = x32.new_tensor(target_sparsity)  # 1-element device scalar (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)

        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 3) Compute std per row on host: std = sqrt(sumsq/F - mean^2) (unbiased=False)
        var = sumsq / F - mean * mean
        # Clamp var to non-negative to guard against tiny negative due to rounding
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # 4) Apply activation via Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)

        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, std, std_multiplier, y,
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
