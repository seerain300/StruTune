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
def compute_inv_ndtri_scalar(out_ptr, p_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) using Abramowitz & Stegun 26.2.23.
    out_ptr: 1-element tensor to store the result
    p_ptr: 1-element tensor containing target_sparsity (float32)
    """
    p = tl.load(p_ptr)  # scalar float
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

    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    # Horner for numerator
    num_low = c1
    num_low = num_low * q_low + c2
    num_low = num_low * q_low + c3
    num_low = num_low * q_low + c4
    num_low = num_low * q_low + c5
    num_low = num_low * q_low + c6
    den_low = d1
    den_low = den_low * q_low + d2
    den_low = den_low * q_low + d3
    den_low = den_low * q_low + d4
    inv_low = num_low / (den_low + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r = q_mid * q_mid
    num_mid = a1
    num_mid = num_mid * r + a2
    num_mid = num_mid * r + a3
    num_mid = num_mid * r + a4
    num_mid = num_mid * r + a5
    num_mid = num_mid * r + a6
    den_mid = b1
    den_mid = den_mid * r + b2
    den_mid = den_mid * r + b3
    den_mid = den_mid * r + b4
    den_mid = den_mid * r + b5
    inv_mid = num_mid * q_mid / (den_mid + 1.0)

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    num_high = c1
    num_high = num_high * q_high + c2
    num_high = num_high * q_high + c3
    num_high = num_high * q_high + c4
    num_high = num_high * q_high + c5
    num_high = num_high * q_high + c6
    den_high = d1
    den_high = den_high * q_high + d2
    den_high = den_high * q_high + d3
    den_high = den_high * q_high + d4
    inv_high = - (num_high / (den_high + 1.0))

    inv = tl.where(mask_low, inv_low, 0.0)
    inv = tl.where(mask_mid, inv_mid, inv)
    inv = tl.where(mask_high, inv_high, inv)

    tl.store(out_ptr, inv)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating: out[row, col] = max(0, x[row, col] - (mean[row] + std[row] * inv))
    Grid: (rows, num_tiles), each program handles one tile of columns for one row.
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load x tile
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)

    # Load scalar inv
    inv = tl.load(inv_ptr)

    # Compute gated output
    threshold = mean + std * inv
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store result
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
      - per-row mean and std (population, unbiased=False) over the last dim
      - inv_norm_cdf(target_sparsity) via A&S approximation
      - output = max(0, inputs - (mean + std * inv_cdf))
    Returns bfloat16 tensor of same shape as input.
    """
    # If no sparsity requested, return inputs unchanged
    if target_sparsity == 0.0:
        return inputs

    # Cast to float32 for computation
    x = inputs
    x_f32 = x.to(torch.float32)
    B, S, N = x_f32.shape
    rows = B * S

    # Flatten to [rows, N] for Triton reduction
    x_2d = x_f32.view(rows, N)

    # 1) Compute per-row mean and std via Triton reduction kernel
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Triton scalar: inverse normal CDF for target_sparsity
    p_tensor = torch.empty(1, device=x.device, dtype=torch.float32)
    p_tensor[0] = float(target_sparsity)
    inv_cdf_tensor = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_tensor, p_tensor, BLOCK_SIZE=1)

    # 3) Triton elementwise gating over 2D tiles
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf_tensor, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape [batch_size, seq_len, intermediate_size]
        assert len(args) == 1, "ModelNew expects a single input tensor"
        return run(*args)