import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d_atomic(x_ptr, sum_ptr, sumsq_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row sum and sum of squares across the last dimension (N).
    Use a 2D grid: axis 0 = row, axis 1 = tile along columns.
    Each program loads one tile (BLOCK_SIZE columns) and atomically adds to the per-row accumulators.
    x_ptr: input flattened as [rows, N] (row-major), contiguous
    sum_ptr/sumsq_ptr: per-row accumulators [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Pointer to the start of the row
    row_base = x_ptr + row_id * N
    # Load tile; out-of-bounds masked elements are set to 0
    x = tl.load(row_base + offs, mask=mask, other=0.0)

    # Partial reductions for this tile
    partial_sum = tl.sum(x)
    partial_sumsq = tl.sum(x * x)

    # Atomically accumulate into per-row totals
    tl.atomic_add(sum_ptr + row_id, partial_sum)
    tl.atomic_add(sumsq_ptr + row_id, partial_sumsq)


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p):
    """
    Triton scalar kernel to compute inverse normal CDF (quantile) for p in (0, 1).
    Uses Abramowitz & Stegun approximation (formula 26.2.23).
    out_ptr: 1-element output tensor
    p: float argument for target sparsity
    """
    # Constants for the approximation
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

    # Work in fp32
    p32 = p
    result = 0.0

    mask_low = p32 < p_low
    # Lower region approximation
    if mask_low:
        q = torch.sqrt(-2.0 * torch.log(p32))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    mask_mid = (p32 >= p_low) & (p32 <= p_high)
    if mask_mid:
        q = p32 - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    mask_high = p32 > p_high
    if mask_high:
        q = torch.sqrt(-2.0 * torch.log(1.0 - p32))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Store to output (1-element tensor)
    tl.store(out_ptr, result)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating for each row and tile of columns:
    y[row, col] = max(0, x[row, col] - (mean[row] + std[row] * inv_cdf))
    2D grid: axis 0 = row, axis 1 = tile along columns
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load the per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    # Compute threshold
    threshold = mean + std * inv_cdf

    # Load tile from x
    x_row_base = x_ptr + row_id * N
    x = tl.load(x_row_base + offs, mask=mask, other=0.0)

    # Apply gating: y = max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store to out
    out_row_base = out_ptr + row_id * N
    tl.store(out_row_base + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-row mean and std, then adaptive threshold per row:
      threshold[row] = mean[row] + std[row] * inv_norm_cdf(target_sparsity)
    and applies gating: y = max(0, x - threshold).
    """
    # Expect a single input tensor with shape [batch_size, seq_len, intermediate_size]
    if inputs.dim() != 3:
        raise RuntimeError("run expects a 3D tensor [batch_size, seq_len, intermediate_size]")

    B, S, N = inputs.shape
    # Cast to float32 for computation
    x = inputs.contiguous().to(torch.float32)
    # Flatten to 2D [rows, N] where rows = B * S
    rows = B * S
    x_2d = x.view(rows, N)

    # 1) Compute per-row sum and sum of squares via Triton atomic reduction
    sum_buf = torch.zeros(rows, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.zeros(rows, device=x.device, dtype=torch.float32)

    BLOCK_SIZE_RS = 1024
    grid_rs = (rows, (N + BLOCK_SIZE_RS - 1) // BLOCK_SIZE_RS)
    reduce_mean_std_2d_atomic[grid_rs](x_2d, sum_buf, sumsq_buf, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # Compute mean and std (population, unbiased=False)
    N_f = float(N)
    mean = sum_buf / N_f
    var = sumsq_buf / N_f - mean * mean
    std = torch.sqrt(var)

    # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

    # 3) Apply gating using 2D Triton kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    grid_gt = (rows, (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT)
    gate_rows_2d[grid_gt](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back to [B, S, N] and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor with shape [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor")
        return run(args[0])


def run(*args):
    return ModelNew()(*args)
