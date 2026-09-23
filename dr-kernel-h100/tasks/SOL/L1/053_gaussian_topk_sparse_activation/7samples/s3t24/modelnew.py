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
    row_id = tl.program_id(axis=0)
    # Accumulate sum and sum of squares in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Compute linear indices for this row
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute mean and store
    mean = sum_val / F
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq_val)


@triton.jit
def var_std_kernel(sumsq_ptr, mean_ptr, std_out_ptr,
                   ROWS: tl.int32, F: tl.int32,
                   BLOCK_SIZE: tl.constexpr):
    """
    Elementwise per row: var = sumsq/F - mean^2; std = sqrt(max(var, 0)).
    """
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sumsq_ptr + row_id)
    mean = tl.load(mean_ptr + row_id)
    var = sumsq / F - mean * mean
    var = tl.maximum(var, 0.0)  # clamp to non-negative
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row_id, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for scalar p (p_in_ptr[0]) using A&S 7.1.26.
    Writes result to p_out_ptr[0].
    """
    # Read p
    p = tl.load(p_in_ptr)  # should be in [0, 1]
    # Clamp p to [eps, 1-eps]
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

    # Regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select by region
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_high = p > p_high
    # Triton does not support Python if; implement with tl.where
    z = tl.where(cond_low, z_low, 0.0)
    z = tl.where(cond_mid, z_mid, z)
    z = tl.where(cond_high, z_high, z)

    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), apply y = max(0, x - (mean[row] + std[row] * multiplier)).
    Inputs:
      - X_ptr: flattened [ROWS*F] float32
      - mean_ptr: [ROWS] float32
      - std_ptr: [ROWS] float32
      - multiplier_ptr: [1] float32
      - Y_ptr: [ROWS*F] float32
    """
    row_id = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    multiplier = tl.load(multiplier_ptr)  # 1-element tensor
    cutoff = mean + std * multiplier

    # Process the row's features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = row_id * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size=1024, num_warps=8, num_stages=2, eps=1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Inputs:
          - inputs: tensor of shape [batch_size, seq_len, intermediate_size], dtype float16/float32
          - target_sparsity: float in [0, 1]
        Output:
          - bfloat16 tensor of same shape as inputs
        """
        B, S, F = inputs.shape
        rows = B * S

        # Cast to float32 for numerical stability in reduction
        x = inputs
        x32 = x.to(torch.float32)

        # 1) Flatten to [rows, F] for row-wise operations
        x_flat = x32.reshape(rows * F)  # contiguous

        # 2) Allocate outputs for stats
        sum_out = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_out = torch.empty(rows, dtype=torch.float32, device=x32.device)
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        std = torch.empty(rows, dtype=torch.float32, device=x32.device)

        # 3) Compute row stats in Triton
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_flat, mean, sumsq_out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 4) Compute scalar inverse normal CDF for target_sparsity in Triton
        p_in = x32.new_tensor(target_sparsity)  # 1-element device scalar
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 5) Compute std per row in Triton: std = sqrt(sumsq/F - mean^2) with clamping
        var_std_kernel[grid_stats](
            sumsq_out, mean, std,
            ROWS=rows, F=F,
            num_warps=4,
            num_stages=2,
        )

        # 6) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        out = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        relu_threshold_kernel[(rows,)](
            x_flat, mean, std, std_multiplier, out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 7) Reshape and cast to bfloat16 to match original output
        y_out = out.view(B, S, F).to(torch.bfloat16)
        return y_out