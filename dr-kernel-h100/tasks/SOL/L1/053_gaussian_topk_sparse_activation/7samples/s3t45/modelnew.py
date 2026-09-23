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
    """
    pid = tl.program_id(axis=0)  # row id
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over feature dimension in tiles
    for offs in range(0, F, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + pid * F + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / F
    # write per-row mean and sum of squares
    tl.store(mean_out_ptr + pid, mean)
    tl.store(sumsq_out_ptr + pid, sum_sq)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile function) for the scalar p_in_ptr[0]
    using Abramowitz and Stegun 7.1.26 approximation. Store result to p_out_ptr[0].
    Clamp p to [eps, 1-eps] to avoid log(0)/log(1) issues.
    """
    p = tl.load(p_in_ptr)
    # clamp to valid range
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)
    # constants
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

    # compute ndtri(p) piecewise
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    result_low = poly_low / denom_low

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    result_mid = poly_mid * q_mid / denom_mid

    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    result_high = -poly_high / denom_high

    res = tl.where(p < p_low, result_low, 0.0)
    res = tl.where(p >= p_high, result_high, res)
    res = tl.where((p >= p_low) & (p <= p_high), result_mid, res)
    tl.store(p_out_ptr, res)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply activation: y = max(0, X - (mean + std * multiplier))
    Per row, load mean and std, compute cutoff, then subtract across features.
    """
    pid = tl.program_id(axis=0)  # row id
    # load per-row scalars
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    multiplier = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * multiplier

    for offs in range(0, F, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + pid * F + cols, mask=mask, other=0.0)
        y = x - cutoff
        # relu: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + pid * F + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-7,
                 block_size: int = 2048, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        1) Compute per-row mean and sum of squares across the last dim (features).
        2) Compute per-row std = sqrt(sumsq/F - mean^2), unbiased=False.
        3) Compute ndtri(target_sparsity) in Triton scalar kernel.
        4) Apply y = max(0, x - (mean + std * multiplier)) in Triton per-row activation kernel.
        5) Return bfloat16 output matching original model.
        """
        assert inputs.is_cuda, "ModelNew requires CUDA tensor input."
        # ensure contiguous float32 for computation
        x = inputs.contiguous()
        x32 = x.to(torch.float32)

        B, S, F = x32.shape
        rows = B * S

        # 1) Compute per-row sum and sum of squares
        sum_vals = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_vals = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32.view(rows, F),
            sum_vals, sumsq_vals,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=8,
            num_stages=2,
        )

        # 2) Compute mean and std per row
        mean = sum_vals / F  # elementwise
        # var = sumsq/F - mean^2  -> unbiased=False (divide by F)
        var = sumsq_vals / F - mean * mean
        std = torch.sqrt(var)  # elementwise on GPU

        # 3) Compute inverse standard normal CDF for scalar target_sparsity in Triton
        p_in = x32.new_tensor(target_sparsity)  # 1-element device scalar (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation in Triton
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