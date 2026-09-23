import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) across the last dimension.
    x_ptr: pointer to input [rows, N] flattened (conceptually)
    mean_ptr/std_ptr: output vectors of length rows (float32)
    N: last dimension size (int32)
    Each program handles one row.
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # For masked loads, use 0.0 so it doesn't affect sums
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    # population std
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri(x_ptr, out_ptr, p: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Compute inv_norm_cdf(p) using Abramowitz & Stegun 7.1.26 approximation.
    Store result into out_ptr[0].
    """
    # Constants (float32)
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

    # We operate in fp32
    p_f = p.to(tl.float32)

    if p_f < p_low:
        q = tl.sqrt(-2.0 * tl.log(p_f))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        val = poly / denom
    elif p_f <= p_high:
        q = p_f - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        inner = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        val = poly * q / (inner * r + 1.0)
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_f))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        val = poly / denom

    # Store into out_ptr[0]
    tl.store(out_ptr, val)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    2D grid elementwise gating:
    For each (row, col_tile), compute y = max(0, x - (mean[row] + std[row] * inv_cdf))
    and write to out.
    x_ptr: [rows, N] flattened conceptual pointer; we index via row_id and col offsets
    mean_ptr/std_ptr: [rows] float32
    inv_ptr: [1] float32 scalar inv_norm_cdf(target_sparsity)
    out_ptr: [rows, N] flattened conceptual pointer
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = (row_id < rows) & (offs < N)

    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar
    threshold = mean + std * inv_cdf

    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    y = tl.maximum(x - threshold, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def _next_power_of_two(n: int, max_pow: int = 1024) -> int:
    # Choose a power-of-two BLOCK_SIZE up to max_pow, not exceeding n if possible
    if n <= 64:
        return 64
    # Compute next power of two
    p = 1
    while p < n and p < max_pow:
        p <<= 1
    # If p > max_pow, pick max_pow
    return min(p, max_pow)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the Gaussian-based top-k sparse activation:
        - Compute per-row mean and std across last dim (feature dim).
        - Compute inv_norm_cdf(target_sparsity) using Triton approximation.
        - Apply gating: y = max(0, x - (mean + std * inv_cdf)), per row.
        Returns tensor cast to bfloat16.
        """
        if target_sparsity == 0.0:
            # No sparsity: return input unchanged in original dtype
            return x

        # Ensure contiguous and compute in float32 for stability
        x_f32 = x.to(torch.float32).contiguous()
        B, S, N = x_f32.shape
        rows = B * S

        # Allocate mean and std vectors
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Reduction kernel: one program per row
        BLOCK_SIZE_RS = _next_power_of_two(N, max_pow=1024)
        grid_rs = (rows,)
        reduce_mean_std[grid_rs](x_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        # Note: p is a scalar float32; Triton supports passing scalar via keyword
        compute_inv_ndtri[(1,)](inv_cdf_buf, p=float(target_sparsity), BLOCK_SIZE=1, num_warps=1)

        # Prepare output and launch 2D gating kernel
        out_flat = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)

        # Choose BLOCK_SIZE for gating; 1024 works well across sizes
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid_gt = (rows, num_tiles)
        gate_rows_2d[grid_gt](x_f32, mean, std, inv_cdf_buf, out_flat, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
