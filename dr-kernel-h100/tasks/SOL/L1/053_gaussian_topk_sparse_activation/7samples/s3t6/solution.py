import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(X, mean_out, var_out, ROWS: tl.int32, F: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and variance over features dimension.
    X: [ROWS, F], float32
    mean_out: [ROWS], float32
    var_out: [ROWS], float32
    """
    row = tl.program_id(0)
    # Accumulators
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over feature dimension in tiles
    for offset in range(0, F, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Each row is contiguous in features, so address = row * F + cols
        x = tl.load(X + row * F + cols, mask=mask, other=0.0)
        # Reduce tile into scalars
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = F
    mean = sum_x / n
    ex2 = sum_x2 / n
    var = ex2 - mean * mean
    # Store results
    tl.store(mean_out + row, mean)
    tl.store(var_out + row, var)


@triton.jit
def _ndtri_rows_kernel(p, std_multiplier_vec, ROWS: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF for each p[i] using A&S 7.1.26 piecewise approximation.
    p: [ROWS], float32, values in (0, 1)
    std_multiplier_vec: [ROWS], float32 output
    """
    i = tl.program_id(0)
    # Clamp to (0,1) to avoid NaNs if p is outside
    p_val = tl.maximum(0.0, tl.minimum(1.0, p[i]))

    # Constants for approximation
    # Lower region
    p_low = 0.02425
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

    # Determine which region
    mask_low = p_val < p_low
    mask_mid = (p_val >= p_low) & (p_val <= (1.0 - p_low))
    mask_high = p_val > (1.0 - p_low)

    q_low = tl.sqrt(-2.0 * tl.log(p_val))
    q_mid = p_val - 0.5
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p_val))

    # Compute polynomial for lower and upper regions (piecewise)
    # Lower region
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / denom_low

    # Central region
    poly_mid = (((((a1 * q_mid * q_mid + a2) * q_mid + a3) * q_mid + a4) * q_mid + a5) * q_mid + a6) * q_mid
    denom_mid = (((((b1 * q_mid * q_mid + b2) * q_mid + b3) * q_mid + b4) * q_mid + b5) * q_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # Upper region
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / denom_high

    # Select region result
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(std_multiplier_vec + i, z)


@triton.jit
def _relu_threshold_kernel(X, mean, std, std_multiplier, Y, ROWS: tl.int32, F: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, x - (mean + std * std_multiplier)) elementwise over rows and features.
    X: [ROWS, F], float32 input
    mean: [ROWS], float32
    std: [ROWS], float32
    std_multiplier: scalar, float32
    Y: [ROWS, F], float32 output
    """
    row = tl.program_id(0)
    for offset in range(0, F, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X + row * F + cols, mask=mask, other=0.0)
        m = tl.load(mean + row)
        s = tl.load(std + row)
        cutoff = m + s * std_multiplier
        y = x - cutoff
        # ReLU
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(Y + row * F + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function:
        - Computes per-row mean and std across features.
        - Computes inverse normal CDF (std_multiplier) via Triton kernel.
        - Applies y = max(0, x - (mean + std * std_multiplier)) in Triton.
        - Returns output as bfloat16 to match original.
        """
        # Ensure contiguous and compute in float32
        x = x.contiguous()
        B, S, F = x.shape
        rows = B * S

        # Create float32 view for Triton
        x_f32 = x.to(torch.float32)

        # Allocate outputs for stats and final activation
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        var = torch.empty(rows, dtype=torch.float32, device=x.device)
        y = torch.empty((rows, F), dtype=torch.float32, device=x.device)

        # 1) Compute row-wise mean and variance in Triton
        _row_stats_kernel[(rows,)](
            x_f32.view(rows, F),
            mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std per row (device-side elementwise op for sqrt)
        std = torch.sqrt(var)

        # 3) Compute std_multiplier per row via Triton kernel
        # Create a 1D tensor of length rows filled with target_sparsity (float32 on device)
        p_rows = torch.full((rows,), float(target_sparsity), dtype=torch.float32, device=x.device)
        std_multiplier_vec = torch.empty(rows, dtype=torch.float32, device=x.device)

        _ndtri_rows_kernel[(rows,)](
            p_rows, std_multiplier_vec,
            ROWS=rows,
            BLOCK_SIZE=1,  # scalar per-lane computation
            num_warps=1,
            num_stages=1,
        )

        # 4) Apply activation in Triton
        _relu_threshold_kernel[(rows,)](
            x_f32.view(rows, F),
            mean, std, std_multiplier_vec, y,
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
