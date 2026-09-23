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
    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over feature dimension in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Linear index for row-major layout
        idx = row * F + cols
        vals = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)
    tl.store(sum_out_ptr + row, sum_val)
    tl.store(sumsq_out_ptr + row, sum_sq)


@triton.jit
def std_kernel(sum_ptr, sumsq_ptr, std_out_ptr,
               ROWS: tl.int32, F: tl.int32,
               BLOCK_SIZE: tl.constexpr):
    """
    Compute std[row] = sqrt(sumsq[row]/F - (sum[row]/F)^2) for each row.
    Implement sqrt via approximation for Triton-only execution.
    """
    row = tl.program_id(0)
    sum_val = tl.load(sum_ptr + row)
    sum_sq = tl.load(sumsq_ptr + row)
    mean = sum_val / F
    var = sum_sq / F - mean * mean
    # Approximate sqrt(var): use Newton's method on y = x^2 - var with initial guess x0 = var
    # x_{n+1} = 0.5 * (x_n + var / x_n)
    x = var  # initial guess
    # A few iterations are sufficient for accuracy
    for _ in range(8):
        x_next = 0.5 * (x + var / x)
        x = x_next
    tl.store(std_out_ptr + row, x)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) for scalar p in (0, 1).
    Uses Abramowitz & Stegun 7.1.26 approximation and clamps to [eps, 1-eps].
    Writes result to p_out_ptr.
    """
    p = tl.load(p_in_ptr)
    # Clamp p to [eps, 1-eps]
    p = tl.maximum(eps, tl.minimum(1.0 - eps, p))
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
    Apply y = max(0, X - (mean + std * multiplier)) per element.
    mean and std are per-row vectors; multiplier is a 1-element tensor.
    """
    row = tl.program_id(0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    multiplier = tl.load(multiplier_ptr)  # scalar
    # Iterate over features in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        cutoff = mean + std * multiplier
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # x: [B, S, F], float32 expected for computation stability
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, S, F = x.shape
        rows = B * S

        # 1) Flatten to [rows, F]
        X = x.reshape(rows, F)

        # 2) Triton reduction: compute sum and sum of squares per row
        sum_out = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_out = torch.empty(rows, dtype=torch.float32, device=x.device)
        row_stats_kernel[(rows,)](
            X, sum_out, sumsq_out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute mean and variance (host-side, simple and fast)
        mean = sum_out / F
        var = sumsq_out / F - mean * mean
        # 4) Triton compute std: implement sqrt via Newton's method
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_kernel[(rows,)](
            sum_out, sumsq_out, std,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,  # not used in this kernel, but kept for consistency
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Triton compute ndtri scalar for target_sparsity
        p_in = x.new_tensor(target_sparsity).contiguous()  # 1-element tensor (no torch.tensor on tensors)
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        multiplier = p_out  # 1-element tensor scalar

        # 6) Triton activation: y = max(0, X - (mean + std * multiplier))
        Y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            X, mean, std, multiplier, Y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 7) Reshape and cast to bfloat16 to match original output
        Y_out = Y.view(B, S, F).to(torch.bfloat16)
        return Y_out


def run(*args):
    return ModelNew()(*args)
