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
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, out_ptr, N, target_sparsity, BLOCK_SIZE: tl.constexpr):
    """
    Triton 2D kernel: elementwise gating over tiles of N for each row.
    Computes threshold = mean + std * inv_norm_cdf(target_sparsity) and writes
    y = max(0, x - threshold) to out_ptr.
    """
    # Load program IDs
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    # Load per-row stats
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)

    # Compute inverse standard normal CDF (Abramowitz & Stegun 7.1.26) for given target_sparsity
    p = target_sparsity  # in (0, 1)
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Piecewise approximation
    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = ((((((-7.784894002430293e-03) * q_low + (-3.223964580411365e-01)) * q_low + (-2.400758277161838e+00)) * q_low + (-2.549732539343734e+00)) * q_low + 4.374664141464968e+00) * q_low + 2.938163982698783e+00) / \
            (((((7.784695709041462e-03) * q_low + 3.224671290700398e-01) * q_low + 2.445134137142996e+00) * q_low + 3.754408661907416e+00) * q_low + 1.0)
    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = ((((((-3.969683028665376e+01) * r_mid + 2.209460984245205e+02) * r_mid + (-2.759285104469687e+02)) * r_mid + 1.383577518672690e+02) * r_mid + (-3.066479806614716e+01)) * r_mid + 2.506628277459239e+00) * q_mid
    denom_mid = ((((((-5.447609879822406e+01) * r_mid + 1.615858368580409e+02) * r_mid + (-1.556989798598866e+02)) * r_mid + 6.680131188771972e+01) * r_mid + (-1.328068155288572e+01)) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid
    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = ((((((-7.784894002430293e-03) * q_up + (-3.223964580411365e-01)) * q_up + (-2.400758277161838e+00)) * q_up + (-2.549732539343734e+00)) * q_up + 4.374664141464968e+00) * q_up + 2.938163982698783e+00) / \
           (((((7.784695709041462e-03) * q_up + 3.224671290700398e-01) * q_up + 2.445134137142996e+00) * q_up + 3.754408661907416e+00) * q_up + 1.0)

    # Select piecewise
    mask_low = p < p_low
    mask_up = p > p_high
    inv_cdf = tl.where(mask_low, z_low, tl.where(mask_up, -z_up, z_mid))

    threshold = mean + std * inv_cdf

    # Process a tile of the row
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def _run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation of the sparsity gating.
    """
    # Convert to float32 for numerical stability in statistics and gating
    x = inputs
    x_f32 = x.to(torch.float32)

    # Flatten to 2D [rows, N] where N = intermediate_size
    N = x_f32.shape[-1]
    rows = x_f32.numel() // N
    x_2d = x_f32.view(rows, N)

    # 1) Triton reduction to compute per-row mean and std (unbiased=False)
    mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Triton elementwise gating over tiles of N (computes inv_norm_cdf inside the kernel)
    out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, out_2d, N, target_sparsity, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

    # Reshape back to original and cast to bfloat16
    out = out_2d.view(*x.shape).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect inputs as a single tensor; target_sparsity as second positional arg (default 0.5)
        if len(args) == 0:
            return None
        inputs = args[0]
        target_sparsity = 0.5
        if len(args) > 1:
            target_sparsity = float(args[1])
        return _run(inputs, target_sparsity)