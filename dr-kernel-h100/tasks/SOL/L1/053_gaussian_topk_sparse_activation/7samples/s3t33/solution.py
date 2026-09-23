import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F, sumsq[row].
    """
    row = tl.program_id(0)
    sum_val = 0.0
    sumsq_val = 0.0

    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / F
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, sumsq_val)


@triton.jit
def var_kernel(X_ptr, mean_ptr, var_out_ptr,
               ROWS: tl.int32, F: tl.int32,
               BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute variance as mean((x - mean)^2) across F.
    Writes var[row].
    """
    row = tl.program_id(0)
    mean = tl.load(mean_ptr + row)
    var_sum = 0.0

    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        diff = x - mean
        var_sum += tl.sum(diff * diff, axis=0)

    var = var_sum / F
    tl.store(var_out_ptr + row, var)


@triton.jit
def std_kernel(var_ptr, std_out_ptr, N_ROWS: tl.int32):
    """
    Elementwise compute std[row] = sqrt(var[row]) for row in [0, N_ROWS).
    """
    row = tl.program_id(0)
    var = tl.load(var_ptr + row)
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for p_in_ptr[0] into p_out_ptr[0]
    using Abramowitz & Stegun 7.1.26 approximation and clamp p to [eps, 1-eps].
    """
    p = tl.load(p_in_ptr)  # scalar
    # Clamp to avoid log(0)/log(1)
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
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        if p > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                     ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        else:
            q = p - 0.5
            r = q * q
            result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                     (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, x - (mean[row] + std[row] * multiplier[0])) elementwise for each row.
    X_ptr is [ROWS, F], Y_ptr is [ROWS*F]. mean_ptr, std_ptr are [ROWS], multiplier_ptr is [1].
    """
    row = tl.program_id(0)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    multiplier = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * multiplier

    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx_in = row * F + cols
        x = tl.load(X_ptr + idx_in, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        idx_out = row * F + cols
        tl.store(Y_ptr + idx_out, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self,
                 block_size: int = 1024,
                 num_warps: int = 8,
                 num_stages: int = 2,
                 eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized implementation of Gaussian-based top-k sparse activation:
        - Compute per-row mean and std across last dim
        - std_multiplier = inverse standard normal CDF (ndtri) of target_sparsity
        - output = max(0, inputs - (mean + std * multiplier)), in bfloat16
        """
        # Ensure contiguous and float32 for stats
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, S, F = x.shape
        rows = B * S

        # 1) Row-wise sum and sum of squares in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)
        row_stats_kernel[(rows,)](
            x.view(rows, F),
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute variance stably as mean((x - mean)^2) in Triton
        var = torch.empty(rows, dtype=torch.float32, device=x.device)
        # Reuse x to compute squared deviations; we need to read x again
        # Launch var_kernel with same tiling
        var_kernel[(rows,)](
            x.view(rows, F),
            mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute std = sqrt(var) in Triton
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_kernel[(rows,)](
            var, std, N_ROWS=rows,
            num_warps=self.num_warps,
            num_stages=1,
        )

        # 4) Compute ndtri(target_sparsity) in Triton (1-element output)
        p_in = x.new_tensor(target_sparsity)  # 1-element device tensor (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 5) Apply activation via Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x.view(rows, F),
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
