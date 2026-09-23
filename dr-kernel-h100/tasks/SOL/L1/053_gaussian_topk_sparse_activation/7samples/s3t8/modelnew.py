import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_sumsq(x_ptr, mean_ptr, sumsq_ptr,
                            rows, F: tl.constexpr, BLOCK: tl.constexpr):
    """
    For each row in [0, rows), compute sum and sum of squares over F features.
    x_ptr: pointer to input [rows, F], float32
    mean_ptr: pointer to output [rows], float32
    sumsq_ptr: pointer to output [rows], float32
    """
    row = tl.program_id(0)
    # Accumulators
    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate over features in tiles of BLOCK
    off = 0
    while off < F:
        cols = off + tl.arange(0, BLOCK)
        # Address for this row and these columns
        # x is contiguous with stride F along features; row starts at row*F
        vals = tl.load(x_ptr + row * F + cols, mask=cols < F, other=0.0)
        total_sum += tl.sum(vals, axis=0)
        total_sumsq += tl.sum(vals * vals, axis=0)
        off += BLOCK

    mean = total_sum / F
    var = total_sumsq / F - mean * mean
    tl.store(mean_ptr + row, mean)
    tl.store(sumsq_ptr + row, var)


@triton.jit
def _sqrt_std(var_ptr, std_ptr, rows, BLOCK: tl.constexpr):
    """
    Compute std = sqrt(var) per row: std_ptr[row] = sqrt(var_ptr[row]).
    var_ptr: [rows] float32
    std_ptr: [rows] float32
    """
    row = tl.program_id(0)
    var = tl.load(var_ptr + row)
    std = tl.sqrt(var)
    tl.store(std_ptr + row, std)


@triton.jit
def _ndtri_scalar(p_ptr, out_ptr, eps=1e-7, BLOCK: tl.constexpr = 256):
    """
    Compute inverse standard normal CDF for the single element at p_ptr[0].
    Writes the result to out_ptr[0].
    Uses Abramowitz & Stegun 7.1.26 approximation with clamping.
    """
    p = tl.load(p_ptr)
    # Clamp to (eps, 1-eps)
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

    # p_low and p_high are scalars; we can precompute them or use constants
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Piecewise evaluation
    if p <= p_low:
        # Lower region
        z = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        poly2 = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        out = poly / poly2
    elif p >= p_high:
        # Upper region
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        poly2 = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        out = -poly / poly2
    else:
        # Central region
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        out = (poly * q) / poly2

    tl.store(out_ptr, out)


@triton.jit
def _apply_activation(x_ptr, mean_ptr, std_ptr, multiplier_ptr, y_ptr,
                       rows, F: tl.constexpr, BLOCK: tl.constexpr):
    """
    For each row, load mean and std, compute cutoff = mean + std * multiplier,
    then write y[row, :] = max(0, x[row, :] - cutoff).
    x_ptr: [rows, F] float32
    mean_ptr, std_ptr: [rows] float32
    multiplier_ptr: [1] float32
    y_ptr: [rows, F] float32
    """
    row = tl.program_id(0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    m = tl.load(multiplier_ptr)  # scalar multiplier
    cutoff = mean + std * m

    off = 0
    while off < F:
        cols = off + tl.arange(0, BLOCK)
        x_vals = tl.load(x_ptr + row * F + cols, mask=cols < F, other=0.0)
        diff = x_vals - cutoff
        # ReLU
        diff = tl.maximum(diff, 0.0)
        tl.store(y_ptr + row * F + cols, diff, mask=cols < F)
        off += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 4):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: Input tensor of shape [batch_size, seq_len, intermediate_size], float32
        target_sparsity: float in (0, 1)
        Returns: bfloat16 tensor of same shape with adaptive sparsity applied via ReLU threshold.
        """
        # Ensure float32 and contiguous
        x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()
        B, S, F = x.shape
        rows = B * S

        # Allocate outputs for mean, sumsq, std
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        # Launch reduction kernel to compute mean and sum of squares per row
        grid_reduce = (rows,)
        _row_reduce_mean_sumsq[grid_reduce](
            x.view(rows, F),
            mean, sumsq,
            rows, F, self.block_size,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Compute var and std via Triton elementwise sqrt kernel
        grid_std = (rows,)
        _sqrt_std[grid_std](
            sumsq, std, rows, self.block_size,
            num_warps=1, num_stages=1
        )

        # Compute ndtri for target_sparsity using Triton
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=x.device)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar[(1,)](
            p_tensor, std_multiplier,
            num_warps=1, num_stages=1
        )

        # Apply activation y = max(0, x - (mean + std * multiplier)) per element
        y = torch.empty_like(x, dtype=torch.float32)
        grid_act = (rows,)
        _apply_activation[grid_act](
            x.view(rows, F), mean, std, std_multiplier, y,
            rows, F, self.block_size,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Cast to bfloat16 to match original
        return y.to(torch.bfloat16)