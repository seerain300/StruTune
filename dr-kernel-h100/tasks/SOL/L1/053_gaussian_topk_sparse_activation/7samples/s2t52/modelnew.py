import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_rows_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    Grid: axis 0 = row id, axis 1 = tile id along N
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Base offset for the row
    row_base = row_id * N
    # Load tile of the row
    x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
    # Accumulate partial sums for this tile
    sum_tile = tl.sum(x, axis=0)
    sumsq_tile = tl.sum(x * x, axis=0)

    # Atomic add into per-row accumulators
    tl.atomic_add(mean_ptr + row_id, sum_tile / N)
    tl.atomic_add(std_ptr + row_id, sumsq_tile / N)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over tiles of N per row:
    out[i, j] = max(0, x[i, j] - (mean[i] + std[i] * inv_cdf))
    Grid: axis 0 = row id, axis 1 = tile id along N
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    row_base = row_id * N

    # Load per-row stats
    mean_val = tl.load(mean_ptr + row_id)
    std_val = tl.load(std_ptr + row_id)
    inv_val = tl.load(inv_ptr)  # scalar inv_norm_cdf

    # Compute threshold per element
    thresh = mean_val + std_val * inv_val
    # Load input tile, compute y = max(0, x - thresh)
    x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
    y = x - thresh
    # ReLU
    y = tl.maximum(y, 0.0)

    # Store result
    tl.store(out_ptr + row_base + offs, y, mask=mask)


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel to compute inverse normal CDF (quantile) using A&S approximation.
    out_ptr: pointer to 1-element float32 output.
    p: float32, value in (0, 1). We use a fixed approximation; we don't call torch.sqrt here.
    """
    # Constants for A&S approximation (formula 26.2.23)
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

    # Determine region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Store scalar result
    tl.store(out_ptr, result)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.

    Computes adaptive sparsity threshold based on input statistics:
    1) per-row mean and std over last dim (unbiased=False)
    2) inv_norm_cdf(target_sparsity) via Triton scalar kernel
    3) elementwise gating: y = max(0, inputs - (mean + std * inv_cdf))

    Returns:
        Sparsified tensor of same shape as input, dtype bfloat16.
    """
    # Handle dtype and device
    x = inputs.contiguous().to(torch.float32)
    B, S, N = x.shape
    rows = B * S

    # 1) Reduce per-row mean and std with Triton
    x_2d = x.view(rows, N)
    mean = torch.zeros(rows, device=x.device, dtype=torch.float32)
    std = torch.zeros(rows, device=x.device, dtype=torch.float32)

    BLOCK_SIZE_RS = 1024
    grid_rs = (rows, (N + BLOCK_SIZE_RS - 1) // BLOCK_SIZE_RS)
    reduce_mean_std_rows_2d[grid_rs](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) in Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

    # 3) Apply gating via Triton 2D kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    grid_gt = (rows, (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT)
    gate_rows_2d[grid_gt](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor with shape [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor")
        return run(args[0])