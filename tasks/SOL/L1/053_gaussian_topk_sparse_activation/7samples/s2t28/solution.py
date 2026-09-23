import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) across the last dimension.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Row offset = row_id * N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_scalar(inv_ptr, p: tl.float32, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for p in (0, 1) using A&S 26.2.23 approximation.
    Writes to inv_ptr[0].
    """
    # A&S constants
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

    # Start with a dummy 0.0
    inv = 0.0
    # Lower region
    mask_low = p < p_low
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        inv = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q = p - 0.5
        r = q * q
        inv = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    mask_high = p > p_high
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        inv = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Store as float32 to inv_ptr[0]
    tl.store(inv_ptr, inv)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows: tl.int32, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating per row and tile of columns:
    For each row i, out[i, col] = max(0, x[i, col] - (mean[i] + std[i] * inv_cdf)),
    where inv_cdf is read from inv_ptr[0] once per program.
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load the row tile
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Load per-row stats and inv_cdf (float32)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar

    # Compute gating threshold and apply
    thr = mean + std * inv_cdf
    y = tl.maximum(x - thr, 0.0)

    # Store
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Returns sparsified tensor of same shape as input.
    """
    # Cast to float32 for numerics
    x = inputs.to(torch.float32)
    B, S, N = x.shape
    rows = B * S

    # 1) Flatten to [rows, N] and compute per-row mean & std (unbiased=False)
    x_2d = x.view(rows, N)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

    # 3) Apply gating using 2D Triton kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
