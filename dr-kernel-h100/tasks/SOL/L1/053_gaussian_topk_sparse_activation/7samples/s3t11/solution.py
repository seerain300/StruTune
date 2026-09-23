import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] and sumsq[row] to mean_out_ptr and sumsq_out_ptr.
    Assumes X_ptr points to a [ROWS, F] logical matrix (flattened). We use
    row * F + col addressing to read elements.
    """
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return

    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over the feature dimension in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Linear index for the row slice
        idx = row_id * F + cols
        # Load values as float32
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Accumulate
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and store outputs
    mean = sum_val / F
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sum_sq)


@triton.jit
def _sqrt_std_kernel(var_ptr, std_ptr, ROWS: tl.int32):
    """
    Elementwise std = sqrt(var) for a [ROWS] vector. Writes std_ptr[i] = sqrt(var_ptr[i]).
    Assumes var_ptr is float32 and std_ptr is float32.
    """
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return
    v = tl.load(var_ptr + row_id)
    s = tl.sqrt(v)
    tl.store(std_ptr + row_id, s)


@triton.jit
def _ndtri_scalar_kernel(p_ptr, out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for p_ptr[0] using Abramowitz & Stegun 7.1.26
    approximation. Writes result to out_ptr[0].
    Clamp p to [eps, 1-eps] to avoid log(0) or log(1).
    """
    p = tl.load(p_ptr)  # scalar float32
    # Clamp p to valid range
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

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

    # Piecewise computation
    # Lower region
    p_low = 0.02425
    p_high = 1.0 - p_low
    mask_low = p < p_low
    # Compute low polynomial: (((c1*t + c2)*t + c3)*t + c4)*t + c5)*t + c6) / ((((d1*t + d2)*t + d3)*t + d4)*t + 1.0)
    t_low = tl.sqrt(-2.0 * tl.log(p))  # p is clamped, so safe
    poly_num = (((((c1 * t_low + c2) * t_low + c3) * t_low + c4) * t_low + c5) * t_low + c6)
    poly_den = ((((d1 * t_low + d2) * t_low + d3) * t_low + d4) * t_low + 1.0)
    z_low = poly_num / poly_den

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    poly_den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_num_mid / poly_den_mid

    # Upper region
    mask_high = p > p_high
    t_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_num_h = (((((c1 * t_high + c2) * t_high + c3) * t_high + c4) * t_high + c5) * t_high + c6)
    poly_den_h = ((((d1 * t_high + d2) * t_high + d3) * t_high + d4) * t_high + 1.0)
    z_high = - (poly_num_h / poly_den_h)

    # Select based on mask
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    Elementwise activation: y = max(0, X - (mean + std * multiplier)).
    X_ptr is [ROWS, F] flattened; mean_ptr, std_ptr, multiplier_ptr are [ROWS].
    Writes to Y_ptr of length ROWS*F (float32).
    """
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return

    # Load per-row scalars
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    m = tl.load(multiplier_ptr)  # scalar

    cutoff = mean + std * m

    # Process the row across F in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 512, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the original run function:
        - Compute per-row mean and std across feature dimension
        - Compute ndtri(target_sparsity) in Triton
        - Apply y = max(0, x - (mean + std * multiplier))
        Returns bfloat16 tensor with same shape as x.
        """
        B, S, F = x.shape
        rows = B * S

        # 1) Flatten and ensure contiguous, compute in float32
        x32 = x.to(torch.float32).contiguous()

        # 2) Compute mean and sumsq per row via Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)
        row_stats_kernel[(rows,)](
            x32, mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute variance and std per row in Triton
        var = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        var[:] = sumsq / F - mean * mean  # fill for kernels
        _sqrt_std_kernel[(rows,)](
            var, std,
            ROWS=rows,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 4) Compute scalar ndtri in Triton (clamp p to [eps, 1-eps])
        p = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p, std_multiplier, self.eps,
            num_warps=1,
            num_stages=1,
        )

        # 5) Apply activation via Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 6) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
