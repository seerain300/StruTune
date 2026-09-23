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
    For each row, compute variance via mean((x - mean)^2) across F features.
    Writes var[row].
    """
    row = tl.program_id(0)
    mean = tl.load(mean_ptr + row)
    var_val = 0.0

    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        diff = x - mean
        var_val += tl.sum(diff * diff, axis=0)

    var = var_val / F
    tl.store(var_out_ptr + row, var)


@triton.jit
def std_kernel(var_ptr, std_out_ptr, N_ROWS: tl.int32):
    """
    Elementwise compute std[row] = sqrt(var[row]) for row in [0, N_ROWS).
    """
    row = tl.program_id(0)
    v = tl.load(var_ptr + row)
    s = tl.sqrt(v)
    tl.store(std_out_ptr + row, s)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for the scalar in p_in_ptr.
    Uses Abramowitz & Stegun 7.1.26 approximation with clamping to [eps, 1-eps].
    Writes result to p_out_ptr.
    """
    # Load probability
    p = tl.load(p_in_ptr)
    # Clamp for numerical stability
    p = tl.maximum(tl.minimum(p, 1.0 - eps), eps)

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
    # Lower region approximation
    mask_low = p < p_low
    # Upper region approximation
    mask_high = p > (1.0 - p_low)

    # Initialize result
    result = tl.zeros_like(p)

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    num = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
    den = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    result = num / den

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    num = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
    den = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    result = -num / den

    # Central region: not needed because we clamp probabilities to outside
    # Store result
    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply elementwise activation: y = max(0, X - (mean + std * multiplier)).
    Each program handles one row.
    """
    row = tl.program_id(0)
    # Load scalars for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    m = tl.load(multiplier_ptr)  # scalar multiplier, 1-element tensor
    cutoff = mean + std * m
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # y = max(0, x - cutoff)
        diff = x - cutoff
        diff = tl.maximum(diff, 0.0)
        tl.store(Y_ptr + idx, diff, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and sum of squares (Triton).
        - Compute variance via mean((x - mean)^2) (Triton).
        - Compute std = sqrt(var) (Triton).
        - Compute ndtri(target_sparsity) scalar in Triton.
        - Apply activation y = max(0, x - (mean + std * multiplier)) in Triton.
        - Return bfloat16 tensor.
        """
        # Ensure float32 for numerical stability
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, S, F = x.shape
        rows = B * S

        # 1) Compute per-row mean and sum of squares in Triton
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

        # 2) Compute variance via squared deviations in Triton
        var = torch.empty(rows, dtype=torch.float32, device=x.device)
        var_kernel[(rows,)](
            x.view(rows, F),
            mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute std elementwise in Triton
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_kernel[(rows,)](
            var, std,
            N_ROWS=rows,
            num_warps=4,   # elementwise kernel
            num_stages=1,
        )

        # 4) Compute ndtri(target_sparsity) in Triton (1-element output)
        p_in = x.new_tensor(target_sparsity)  # 1-element device tensor (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, 1e-7,
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