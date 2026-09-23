import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, var_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features,
    then mean = sum/F and variance = sumsq/F - mean^2.
    Writes mean[row] and var[row] to mean_out_ptr and var_out_ptr.
    """
    row = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over feature dimension in tiles of BLOCK_SIZE
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + row * F + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / F
    var = sum_sq / F - mean * mean  # unbiased=False
    tl.store(mean_out_ptr + row, mean)
    tl.store(var_out_ptr + row, var)


@triton.jit
def sqrt_var_kernel(var_in_ptr, std_out_ptr,
                    ROWS: tl.int32,
                    BLOCK_SIZE: tl.constexpr):
    """
    Elementwise std = sqrt(var) for a 1D vector of length ROWS.
    """
    row = tl.program_id(0)
    var = tl.load(var_in_ptr + row)
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for the scalar p_in_ptr[0]
    using Abramowitz & Stegun 7.1.26 approximation, write to p_out_ptr[0].
    Clamp input to [eps, 1-eps] to avoid log(0)/log(1).
    """
    p = tl.load(p_in_ptr)
    # Clamp p to valid range
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

    # Lower region
    p_low = 0.02425
    q = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den_low = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    nd_low = poly_low / den_low

    # Central region
    p_mid = 0.5
    q = p - p_mid
    r = q * q
    poly_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    nd_mid = poly_mid / den_mid

    # Upper region
    p_high = 1.0 - p_low
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den_up = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    nd_up = -poly_up / den_up

    # Combine using region masks
    low_mask = p < p_low
    mid_mask = (p >= p_low) & (p <= p_high)
    high_mask = p > p_high

    # Default to mid approximation
    nd_result = nd_mid
    # Select region results
    nd_result = tl.where(low_mask, nd_low, nd_result)
    nd_result = tl.where(high_mask, nd_up, nd_result)

    tl.store(p_out_ptr, nd_result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, X - (mean + std * multiplier)) across flattened [ROWS, F].
    mean_ptr, std_ptr are length ROWS; multiplier_ptr is a 1-element tensor.
    """
    row = tl.program_id(0)
    # Compute cutoff for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    multiplier = tl.load(multiplier_ptr)  # 1-element tensor
    cutoff = mean + std * multiplier
    # Iterate over features in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + row * F + cols, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + row * F + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-7, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of:
          y = max(0, x - (mean + std * ndtri(target_sparsity)))
        with std computed as unbiased=False: std = sqrt(E[x^2] - mean^2).
        Returns y in bfloat16.
        """
        # Ensure input is on CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x32 = x.to(torch.float32).contiguous()

        # Flatten to [rows, F]
        B, S, F = x32.shape
        rows = B * S
        X = x32.view(rows, F)

        # 1) Compute mean and var per row in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        var = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            X, mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std = sqrt(var) in Triton (elementwise)
        std = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sqrt_var_kernel[(rows,)](
            var, std,
            ROWS=rows,
            BLOCK_SIZE=1,  # single element per program
            num_warps=2,
            num_stages=1,
        )

        # 3) Compute scalar ndtri(target_sparsity) in Triton
        p_in = x32.new_tensor(target_sparsity).contiguous()  # 1-element tensor, no torch.tensor on tensors
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Apply activation in Triton: y = max(0, X - (mean + std * multiplier))
        Y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        relu_threshold_kernel[(rows,)](
            X, mean, std, std_multiplier, Y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        Y_out = Y.view(B, S, F).to(torch.bfloat16)
        return Y_out


def run(*args):
    return ModelNew()(*args)
