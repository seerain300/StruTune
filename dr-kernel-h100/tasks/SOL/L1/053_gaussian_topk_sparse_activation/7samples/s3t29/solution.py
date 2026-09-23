import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_sum_sumsq_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                                ROWS: tl.int32, F: tl.int32,
                                BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Write sum[row], sumsq[row] to sum_out_ptr and sumsq_out_ptr (float32).
    """
    row = tl.program_id(0)
    # Compute sum and sumsq over the F features for this row
    sum_ = 0.0
    sumsq_ = 0.0
    for col in range(0, F, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        x = tl.load(X_ptr + row * F + offs, mask=mask, other=0.0)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
    tl.store(sum_out_ptr + row, sum_)
    tl.store(sumsq_out_ptr + row, sumsq_)


@triton.jit
def std_kernel(sum_ptr, sumsq_ptr, std_out_ptr,
               ROWS: tl.int32, F: tl.int32,
               BLOCK_SIZE: tl.constexpr):
    """
    Compute std[row] = sqrt(sumsq[row]/F - sum[row]/F) for each row.
    Writes to std_out_ptr (float32).
    """
    row = tl.program_id(0)
    sum_row = tl.load(sum_ptr + row)
    sumsq_row = tl.load(sumsq_ptr + row)
    mean = sum_row / F
    var = (sumsq_row / F) - mean * mean
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def relu_threshold_kernel(X_ptr, Y_ptr,
                           mean_ptr, std_ptr, p_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, x - (mean + std * multiplier)) for each element in X_ptr,
    with mean/std broadcast per row and multiplier read from p_ptr[0].
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    for col in range(0, F, BLOCK_SIZE):
        idx = col + offs
        mask = idx < F
        x = tl.load(X_ptr + row * F + idx, mask=mask, other=0.0)
        mean_row = tl.load(mean_ptr + row)
        std_row = tl.load(std_ptr + row)
        multiplier = tl.load(p_ptr)  # scalar
        cutoff = mean_row + std_row * multiplier
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(Y_ptr + row * F + idx, y, mask=mask)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for p_in_ptr[0] using A&S 7.1.26.
    Write result to p_out_ptr[0] (float32). Clamp p to [eps, 1-eps] to avoid edge cases.
    """
    p = tl.load(p_in_ptr)
    # Clamp p to [eps, 1-eps]
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
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select by region
    region_low = p < p_low
    region_mid = (p >= p_low) & (p <= p_high)
    region_high = p > p_high

    z = tl.where(region_low, z_low, 0.0)
    z = tl.where(region_mid, z_mid, z)
    z = tl.where(region_high, z_high, z)

    tl.store(p_out_ptr, z)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 2048, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are float32 and contiguous
        x = inputs
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        # Flatten to [rows, F], rows = B * S
        B, S, F = x.shape
        rows = B * S
        x2d = x.view(rows, F)

        # 1) Compute per-row sum and sumsq in Triton
        sum_buf = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(rows, dtype=torch.float32, device=x.device)
        row_stats_sum_sumsq_kernel[(rows,)](
            x2d,
            sum_buf, sumsq_buf,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std per row in Triton
        std_buf = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_kernel[(rows,)](
            sum_buf, sumsq_buf, std_buf,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute inverse normal CDF for scalar target_sparsity using Triton
        p_in = x.new_tensor(target_sparsity)  # 1-element device scalar
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x2d, y,
            sum_buf / F, std_buf, std_multiplier,
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
