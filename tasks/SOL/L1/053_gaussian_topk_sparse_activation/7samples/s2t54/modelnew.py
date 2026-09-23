import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_rows_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns
    """
    row_id = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    start = col_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Base pointer for this row
    row_base = x_ptr + row_id * N
    # Load a tile of the row
    x = tl.load(row_base + offs, mask=mask, other=0.0)

    # Partial sums for this tile
    sum_tile = tl.sum(x, axis=0)
    sumsq_tile = tl.sum(x * x, axis=0)

    # Accumulate into per-row scalars
    tl.atomic_add(mean_ptr + row_id, sum_tile / N)
    tl.atomic_add(std_ptr + row_id, sumsq_tile / N)


@triton.jit
def compute_inv_ndtri_scalar(inv_ptr, p: tl.float32):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) for p in (0,1).
    Uses Abramowitz & Stegun approximation (formula 26.2.23).
    Writes result to inv_ptr[0].
    """
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

    # Lower region: x = sqrt(-2 ln p)
    # q = sqrt(-2 ln p)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    # Polynomial for low region
    poly_low = c1 * q_low + c2
    poly_low = poly_low * q_low + c3
    poly_low = poly_low * q_low + c4
    poly_low = poly_low * q_low + c5
    poly_low = poly_low * q_low + c6

    den_low = d1 * q_low + d2
    den_low = den_low * q_low + d3
    den_low = den_low * q_low + d4
    den_low = den_low * q_low + 1.0

    x_low = poly_low / den_low

    # Central region: x = (p - 0.5) / q, q = p - 0.5
    # For this region, q = p - 0.5, but use general formula with r = q^2
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = a1 * r_mid + a2
    poly_mid = poly_mid * r_mid + a3
    poly_mid = poly_mid * r_mid + a4
    poly_mid = poly_mid * r_mid + a5
    poly_mid = poly_mid * r_mid + a6

    den_mid = b1 * r_mid + b2
    den_mid = den_mid * r_mid + b3
    den_mid = den_mid * r_mid + b4
    den_mid = den_mid * r_mid + b5

    x_mid = poly_mid / den_mid

    # Upper region: x = -sqrt(-2 ln(1 - p))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = c1 * q_high + c2
    poly_high = poly_high * q_high + c3
    poly_high = poly_high * q_high + c4
    poly_high = poly_high * q_high + c5
    poly_high = poly_high * q_high + c6

    den_high = d1 * q_high + d2
    den_high = den_high * q_high + d3
    den_high = den_high * q_high + d4
    den_high = den_high * q_high + 1.0

    x_high = -poly_high / den_high

    # Select based on p
    # Triton doesn't support dynamic selection easily; use masks and where
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = ~mask_low & ~mask_mid

    # Default to mid, then override
    x = x_mid
    x = tl.where(mask_low, x_low, x)
    x = tl.where(mask_high, x_high, x)

    # Store to inv_ptr[0]
    tl.store(inv_ptr, x)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating kernel over tiles of N per row.
    out_ptr: [rows, N], float32
    x_ptr: [rows, N], float32
    mean_ptr/std_ptr: [rows], float32
    inv_ptr: [1], float32, holds inv_norm_cdf(target_sparsity)
    """
    row_id = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    start = col_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row base pointers
    x_row = x_ptr + row_id * N
    out_row = out_ptr + row_id * N

    # Load data for this tile
    x = tl.load(x_row + offs, mask=mask, other=0.0)
    # Load per-row stats and inv_cdf
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv = tl.load(inv_ptr)  # scalar

    # Compute threshold per element: mean + std * inv
    threshold = mean + std * inv
    # Gate: y = max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store results
    tl.store(out_row + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-based implementation of the Gaussian top-k sparse activation.
    Computes adaptive threshold: mean + std * inv_cdf(target_sparsity),
    applies gating: y = max(0, x - threshold).
    """
    # Ensure inputs are contiguous and float32 for computation
    x = inputs.to(torch.float32).contiguous()
    B, S, N = x.shape
    rows = B * S

    # 1) Flatten to 2D [rows, N]
    x_2d = x.view(rows, N)

    # Allocate per-row mean and std (float32)
    mean = torch.zeros(rows, device=x.device, dtype=torch.float32)
    std = torch.zeros(rows, device=x.device, dtype=torch.float32)

    # 2) Triton reduction: per-row mean and population std (unbiased=False)
    BLOCK_SIZE_RS = 256  # tile size along N; 256 is a safe, vectorized choice
    grid_rs = (rows, (N + BLOCK_SIZE_RS - 1) // BLOCK_SIZE_RS)
    reduce_mean_std_rows_2d[grid_rs](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 3) Triton scalar kernel: inv_norm_cdf(target_sparsity)
    inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

    # 4) Triton elementwise gating over tiles
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    grid_gt = (rows, (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT)
    gate_rows_2d[grid_gt](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # 5) Reshape back and cast to bfloat16
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor with shape [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor")
        return run(args[0])