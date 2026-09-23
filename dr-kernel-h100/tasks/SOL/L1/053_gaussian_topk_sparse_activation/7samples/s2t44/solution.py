import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)

    # Accumulators as scalars
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over the row in chunks. For typical large N, we will iterate.
    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # x is laid out as [rows, N] contiguous; row offset is row_id * N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) at probability p
    using Abramowitz and Stegun approximation (formula 26.2.23).
    Writes result to out_ptr[0] as float32.
    """
    # Coefficients (float32)
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

    # Compute z = sqrt(2 * |log p|) depending on region
    # Lower region
    if p > p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        y = poly / poly2
        out_val = -y
    else:
        # Upper region
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        y = poly / poly2
        out_val = y

    # Central region
    if (p >= p_low) & (p <= p_high):
        q = p - 0.5
        r = q * q
        poly3 = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly4 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        out_val = (((poly3 * q) / poly4))

    tl.store(out_ptr, out_val)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr,
                 rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton 2D kernel: apply gating per row over tiles of columns.
    y = max(0, x - (mean[row] + std[row] * inv_cdf))
    x_ptr: [rows, N], float32
    mean_ptr/std_ptr: [rows], float32
    inv_cdf_ptr: [1], float32
    out_ptr: [rows, N], float32
    rows, N: int32
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row data
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    # Load per-row stats
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    # Compute threshold and gating
    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run(inputs, target_sparsity):
        - Computes per-row mean and std across last dim.
        - Computes inv_norm_cdf(target_sparsity) via Triton scalar kernel.
        - Applies gating y = max(0, x - (mean + std * inv_cdf)) using a Triton 2D kernel.
        Returns output cast to bfloat16.
        """
        # Ensure float32 for computation
        x_f32 = x.to(torch.float32)

        # Flatten leading dims into rows: rows = B * S, N = intermediate_size
        N = x_f32.shape[-1]
        rows = x_f32.numel() // N
        x_2d = x_f32.view(rows, N)

        # 1) Triton reduction: mean and population std (unbiased=False)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Triton scalar: inv_norm_cdf(target_sparsity)
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

        # 3) Triton 2D gating over tiles
        out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back to original and cast to bfloat16
        out = out_2d.view(*x_f32.shape).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
