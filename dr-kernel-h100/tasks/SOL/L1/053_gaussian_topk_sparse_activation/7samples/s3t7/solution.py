import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_sum_sq_kernel(x_ptr, mean_ptr, sum_sq_ptr,
                              ROWS: tl.constexpr, F: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr):
    """
    For each row in [ROWS], compute sum and sum of squares across F features.
    Writes:
      mean_ptr[row] = sum / F
      sum_sq_ptr[row] = sum of squares
    """
    row = tl.program_id(0)
    # Accumulators
    total_sum = 0.0
    total_sq = 0.0

    # Tile over feature dimension
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Addressing: row offset is row * F, feature offset is cols
        ptrs = x_ptr + row * F + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        total_sum += tl.sum(vals, axis=0)
        total_sq += tl.sum(vals * vals, axis=0)

    # Compute mean and sum of squares
    mean = total_sum / F
    sum_sq = total_sq  # returned separately to avoid atomic ops

    # Store results (fp32)
    tl.store(mean_ptr + row, mean)
    tl.store(sum_sq_ptr + row, sum_sq)


@triton.jit
def _sqrt_std_kernel(var_ptr, std_ptr, ROWS: tl.constexpr):
    """
    Elementwise std = sqrt(var) over ROWS.
    var_ptr: [ROWS] float32
    std_ptr: [ROWS] float32
    """
    row = tl.program_id(0)
    v = tl.load(var_ptr + row)
    # Triton supports sqrt on float32; clamp to avoid potential negatives
    v = tl.maximum(v, 0.0)
    s = tl.sqrt(v)
    tl.store(std_ptr + row, s)


@triton.jit
def _ndtri_scalar_kernel(p_ptr, out_ptr, eps: tl.constexpr):
    """
    Compute inverse standard normal CDF for scalar p using A&S 7.1.26.
    p_ptr: 1-element tensor (float32)
    out_ptr: 1-element tensor (float32) to store ndtri(p)
    eps: small positive constant to avoid log(0) or log(1) at extremes
    """
    # Load p (scalar as vector)
    p = tl.load(p_ptr)
    # Clamp to (eps, 1-eps)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants (float32)
    a1 = -3.9696830e+01
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
    p_high = 0.97575

    # Piecewise evaluation
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region (p ~ 0.5)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select based on p
    mask_low = p < p_low
    mask_high = p > p_high
    # Triton supports elementwise selection
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_high, z_high, 0.0) + tl.where(~(mask_low | mask_high), z_mid, 0.0)

    # Store scalar result
    tl.store(out_ptr, z)


@triton.jit
def _relu_threshold_kernel(x_ptr, mean_ptr, std_ptr, multiplier, out_ptr,
                            ROWS: tl.constexpr, F: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    """
    Apply ReLU threshold per element:
      y = max(0, x - (mean + std * multiplier))
    x_ptr: [ROWS, F] flattened pointer (rows contiguous)
    mean_ptr: [ROWS] float32
    std_ptr: [ROWS] float32
    multiplier: scalar float32
    out_ptr: [ROWS, F] flattened float32
    """
    row = tl.program_id(0)
    cutoff = tl.load(mean_ptr + row) + tl.load(std_ptr + row) * multiplier

    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x_vals = tl.load(x_ptr + row * F + cols, mask=mask, other=0.0)
        # y = max(0, x - cutoff)
        y_vals = x_vals - cutoff
        y_vals = tl.maximum(y_vals, 0.0)
        tl.store(out_ptr + row * F + cols, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the original run logic:
        - Compute per-row mean and std (float32)
        - Compute std_multiplier via Triton ndtri kernel
        - Apply ReLU(x - (mean + std * multiplier)) elementwise, write float32, cast to bfloat16
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # Work in float32 for numerical stability
        x_f32 = x.to(torch.float32)
        B, S, F = x_f32.shape
        rows = B * S

        # Allocate outputs for stats and activation
        mean = torch.empty(rows, dtype=torch.float32, device=x_f32.device)
        sum_sq = torch.empty(rows, dtype=torch.float32, device=x_f32.device)
        std = torch.empty(rows, dtype=torch.float32, device=x_f32.device)

        # 1) Compute row-wise mean and sum of squares in Triton
        grid_stats = (rows,)
        _row_stats_sum_sq_kernel[grid_stats](
            x_f32, mean, sum_sq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute variance and std in Triton (elementwise sqrt over rows)
        _sqrt_std_kernel[(rows,)](
            sum_sq, std,
            ROWS=rows,
            num_warps=1,
            num_stages=1,
        )

        # 3) Compute std multiplier (inverse normal CDF) in Triton (scalar op)
        p_tensor = torch.empty(1, dtype=torch.float32, device=x_f32.device)
        p_tensor[0] = float(target_sparsity)  # host-side set, but still only one tensor
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        _ndtri_scalar_kernel[(1,)](
            p_tensor, std_multiplier,
            eps=self.eps,
            num_warps=1,
            num_stages=1,
        )

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x_f32.device)

        _relu_threshold_kernel[(rows,)](
            x_f32.view(rows * F),  # flattened pointer to [rows, F] logically
            mean, std, std_multiplier, y,
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
