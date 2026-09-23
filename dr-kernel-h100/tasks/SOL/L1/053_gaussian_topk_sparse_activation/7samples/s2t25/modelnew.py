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
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Load a chunk; x_ptr points to the start of row row_id, which is at offset row_id * N in the flattened view
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
def compute_inv_ndtri_scalar(out_ptr, p: tl.float32):
    """
    Compute inverse standard normal CDF (quantile function) for p in (0, 1)
    using Abramowitz & Stegun 7.1.26 approximation.
    Writes a single float32 result to out_ptr[0].
    """
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower region approximation
        q = tl.sqrt(-2.0 * tl.log(p))
        nd = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        # Upper region approximation
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        nd = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # Central region
        q = p - 0.5
        r = q * q
        nd = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
             (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(out_ptr, nd)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr,
                 rows: tl.int32, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over a 2D input [rows, N]:
    y[i, j] = max(0, x[i, j] - (mean[i] + std[i] * inv_ptr[0]))
    x_ptr: [rows, N] flattened contiguous
    mean_ptr/std_ptr: [rows] float32
    inv_ptr: [1] float32 (scalar)
    out_ptr: [rows, N] float32
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load the row chunk
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Load per-row stats
    mean_val = tl.load(mean_ptr + row_id)
    std_val = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar

    threshold = mean_val + std_val * inv_cdf
    y = tl.maximum(x - threshold, 0.0)

    # Store result
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA tensors and compute in fp32
        if not input.is_cuda:
            input = input.to('cuda')
        x = input.contiguous().to(torch.float32)

        # Flatten to 2D [rows, N] where N is last dimension
        *prefix, N = x.shape
        rows = 1
        for d in prefix:
            rows *= d

        x_2d = x.view(rows, N)

        # 1) Compute per-row mean and std (population, unbiased=False)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

        # 3) Apply gating via 2D Triton kernel over tiles of N
        out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back to original and cast to bfloat16
        out = out_2d.view(*x.shape).to(torch.bfloat16)
        return out