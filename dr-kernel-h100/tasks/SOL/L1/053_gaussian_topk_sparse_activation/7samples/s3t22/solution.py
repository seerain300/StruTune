import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F to mean_out_ptr and sumsq[row] to sumsq_out_ptr.
    """
    row_id = tl.program_id(axis=0)
    # Accumulators
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over feature dimension in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = row_id * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        # Accumulate
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute mean
    mean = sum_val / F
    # Store outputs
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq_val)


@triton.jit
def var_std_kernel(sumsq_ptr, mean_ptr, std_ptr,
                   ROWS: tl.int32, F: tl.int32):
    """
    For each row (0..ROWS-1), compute var = sumsq/F - mean^2 and std = sqrt(var).
    Writes std per row to std_ptr.
    """
    row_id = tl.program_id(axis=0)
    sumsq = tl.load(sumsq_ptr + row_id)
    mean = tl.load(mean_ptr + row_id)
    var = sumsq / F - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(std_ptr + row_id, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile function) for the scalar
    p = p_in_ptr[0] using Abramowitz & Stegun 7.1.26 approximation.
    Writes result to p_out_ptr[0].
    """
    # Load input probability
    p = tl.load(p_in_ptr)
    # Clamp to [eps, 1-eps] for numerical stability
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants for A&S 7.1.26 approximation
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
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select region
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_up = p > p_high

    # Combine results
    # Note: Triton doesn't support Python's max like numpy; use where
    z = tl.zeros_like(p)
    z = tl.where(mask_low, z_low, z)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_up, z_up, z)

    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Out_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), apply activation:
      y = max(0, X - (mean + std * multiplier))
    Writes to Out_ptr[row * F : (row + 1) * F].
    """
    row_id = tl.program_id(axis=0)
    base = row_id * F

    # Load per-row scalars
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    multiplier = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * multiplier

    # Iterate over feature dimension in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = base + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(Out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        inputs: Tensor of shape [batch_size, seq_len, intermediate_size], dtype float32 or float16/bfloat16.
        target_sparsity: float in (0, 1), target sparsity level for Gaussian threshold.
        Returns: Tensor of shape [batch_size, seq_len, intermediate_size], dtype bfloat16.
        """
        # Early exit: no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure float32 for numerical stability
        x = inputs.to(torch.float32)
        B, S, F = x.shape
        rows = B * S

        # 1) Flatten to [rows, F]
        x_flat = x.view(rows, F)

        # Allocate outputs for stats
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)

        # 2) Compute row stats in Triton
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_flat,
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Compute scalar ndtri(target_sparsity) in Triton -> 1-element tensor
        p_in = x_flat.new_tensor(target_sparsity)  # 1-element device tensor
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Compute std per row in Triton: std = sqrt(sumsq/F - mean^2)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        var_std_kernel[grid_stats](
            sumsq, mean, std,
            ROWS=rows, F=F,
            num_warps=4,
            num_stages=2,
        )

        # 5) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        out = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        grid_act = (rows,)
        relu_threshold_kernel[grid_act](
            x_flat, mean, std, std_multiplier, out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 6) Reshape and cast to bfloat16 to match original output
        y_out = out.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
