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
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel to compute inverse of standard normal CDF for a single p in (0,1).
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    Writes result to out_ptr[0] as float32.
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

    # We only compute for one scalar
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        tl.store(out_ptr, z)
        return

    # Central region
    if p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        tl.store(out_ptr, z)
        return

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
        ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    tl.store(out_ptr, z)
    return


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating kernel: per row, tile over columns.
    out = max(0, x - (mean + std * inv_cdf))
    x_ptr: [rows, N] input
    mean_ptr/std_ptr: [rows]
    inv_ptr: [1] scalar result from compute_inv_ndtri_scalar
    out_ptr: [rows, N] output
    rows: number of rows (int32)
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv = tl.load(inv_ptr)  # scalar

    # Load input tile
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    # Compute threshold and gating
    threshold = mean + std * inv
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
      1) Per-row mean and std across last dim (feature dimension)
      2) inv_norm_cdf(target_sparsity) via A&S approximation
      3) Apply gating: y = max(0, inputs - (mean + std * inv_cdf))
    Returns bfloat16 tensor of same shape.
    """
    # Only one input tensor expected: [B, S, N]
    assert inputs.is_cuda, "ModelNew.forward requires CUDA tensors."
    B, S, N = inputs.shape
    rows = B * S

    # Cast to float32 for computation
    x_f32 = inputs.contiguous().to(torch.float32)
    x_2d = x_f32.view(rows, N)

    # 1) Compute per-row mean and std (population, unbiased=False)
    mean = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=x_2d.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, p=float(target_sparsity), BLOCK_SIZE=1)

    # 3) Apply gating using 2D Triton kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input shaped [batch_size, seq_len, intermediate_size]
        if len(args) == 1:
            x = args[0]
        else:
            x = args[0]
        return run(x, 0.01)  # target_sparsity from the original signature