import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum / F to mean_out_ptr and sumsq[row] to sumsq_out_ptr.
    """
    row_id = tl.program_id(0)
    # Each program handles one row; loop over feature dimension in tiles
    total_sum = 0.0
    total_sumsq = 0.0
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        row_base = row_id * F
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=0.0)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)
    mean = total_sum / F
    sumsq = total_sumsq  # we will compute var = sumsq/F - mean^2 on host
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) of p_in_ptr[0] using Abramowitz and Stegun 7.1.26
    approximation and write to p_out_ptr[0].
    """
    p = tl.load(p_in_ptr)
    # Clamp to avoid log(0)/log(1) issues
    p = tl.maximum(tl.minimum(p, 1.0 - eps), eps)

    # Constants for the approximation
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
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly / den

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select region based on p
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_high = p > p_high

    # Triton doesn't support branching on scalars with if; use where to blend
    z = tl.where(cond_low, z_low, 0.0) + tl.where(cond_mid, z_mid, 0.0) + tl.where(cond_high, z_high, 0.0)
    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), apply y = max(0, x - (mean[row] + std[row] * multiplier)),
    with x in X_ptr viewed as [ROWS, F] and Y_ptr as [ROWS*F].
    """
    row_id = tl.program_id(0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    mult = tl.load(multiplier_ptr)  # scalar tensor
    cutoff = mean + std * mult

    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        row_base = row_id * F
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=0.0)
        diff = x - cutoff
        # ReLU: max(0, diff)
        y = tl.where(diff > 0.0, diff, 0.0)
        tl.store(Y_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            # Match original dtype; output is bfloat16 in the original run
            return inputs.to(torch.bfloat16)

        # Flatten to [rows, F] where rows = batch_size * seq_len
        x = inputs
        B, S, F = x.shape
        rows = B * S
        x32 = x.to(torch.float32)

        # 1) Compute per-row sum and sum of squares via Triton
        sum_ptr = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_ptr = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32, sum_ptr, sumsq_ptr,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute mean and variance (unbiased=False) and std per row
        # mean = sum / F
        mean = sum_ptr / F
        # var = sumsq / F - mean^2
        var = sumsq_ptr / F - mean * mean
        # Clamp var to >= 0 to avoid tiny negative due to rounding
        var = tl.maximum(var, 0.0)  # Note: we cannot store into var tensor; create std as new tensor
        std = torch.sqrt(var)

        # 3) Compute inverse CDF for target_sparsity in Triton (scalar)
        p_in = x32.new_tensor(target_sparsity)  # 1-element device tensor (float32)
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
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out