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
    # Accumulators as scalars
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over columns in tiles of BLOCK_SIZE
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Linear index for row and cols
        idx = row * F + cols
        # Load values; masked lanes use 0.0
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Accumulate
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / F
    # Write results for this row
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, sumsq_val)


@triton.jit
def std_kernel(sumsq_ptr, mean_ptr, std_out_ptr,
               N_ROWS: tl.int32,
               F: tl.int32):
    """
    Elementwise compute std[row] = sqrt(sumsq[row]/F - mean[row]^2) for row in [0, N_ROWS).
    Writes results to std_out_ptr.
    """
    row = tl.program_id(0)
    sumsq = tl.load(sumsq_ptr + row)
    mean = tl.load(mean_ptr + row)
    var = sumsq / F - mean * mean
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) of scalar p_in (0 < p_in < 1) using
    Abramowitz & Stegun 7.1.26 approximation. Store result in p_out_ptr[0].
    Clamp p_in to [eps, 1-eps] to avoid log(0)/log(1) issues.
    """
    # Load p
    p = tl.load(p_in_ptr)
    # Clamp
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
    # q = sqrt(-2 * log(p))
    q = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    # r = (p - 0.5)
    r = p - 0.5
    result_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * r / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q2 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_up = -(((((c1 * q2 + c2) * q2 + c3) * q2 + c4) * q2 + c5) * q2 + c6) / \
                ((((d1 * q2 + d2) * q2 + d3) * q2 + d4) * q2 + 1.0)

    # Select by piecewise
    # Triton supports where selection
    out = tl.where(p < p_low, result_low, 0.0)
    out = tl.where(p > p_high, result_up, out)
    # For p in [p_low, p_high], overwrite mid
    out = tl.where((p >= p_low) & (p <= p_high), result_mid, out)

    tl.store(p_out_ptr, out)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply activation: for each row r in [0, ROWS), compute cutoff = mean[r] + std[r] * multiplier,
    then write Y[row*F + cols] = max(0, X[row*F + cols] - cutoff).
    """
    row = tl.program_id(0)
    # Load row scalars
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    mult = tl.load(multiplier_ptr)  # scalar multiplier
    cutoff = mean + std * mult

    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized replacement for the original Model.
    All tensor computations happen inside Triton kernels. Forward only allocates tensors,
    sets grid sizes, and launches Triton kernels.
    """
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        inputs: Tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: float in (0, 1)
        returns: Tensor of shape [batch_size, seq_len, intermediate_size] in bfloat16
        """
        # Ensure dtype is float32 for numerical stability and Triton compatibility
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

        # 2) Compute std elementwise in Triton
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_kernel[(rows,)](
            sumsq, mean, std,
            N_ROWS=rows, F=F,
            num_warps=4,  # small elementwise kernel
            num_stages=1,
        )

        # 3) Compute ndtri(target_sparsity) in Triton (1-element output)
        p_in = x.new_tensor(target_sparsity)  # 1-element device tensor (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation via Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x.view(rows, F),
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out