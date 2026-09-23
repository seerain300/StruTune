import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

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
    Triton scalar kernel: compute inverse standard normal CDF for scalar p in (0, 1).
    Uses Abramowitz and Stegun approximation (26.2.23).
    Writes to out_ptr[0].
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

    # Compute q and apply piecewise approximation
    # This is a scalar kernel, so we operate on out_ptr[0]
    p0 = p[0]
    result = 0.0

    mask_low = p0 < p_low
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p0))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    mask_mid = (p0 >= p_low) & (p0 <= p_high)
    if mask_mid:
        q = p0 - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    mask_high = p0 > p_high
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p0))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(out_ptr, result)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr,
                 rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating kernel over [rows, N]:
    For each row, load mean[row], std[row], inv_cdf, compute threshold,
    and write y = max(0, x - threshold) to out_ptr[row, :].
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input row slice
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Load per-row stats
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar

    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU equivalent

    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        1) Compute per-row mean and std across feature dim (last dim).
        2) Compute inv_norm_cdf(target_sparsity) using A&S approximation.
        3) Apply gating: y = max(0, x - (mean + std * inv_cdf)).
        Returns bfloat16 tensor of same shape as inputs.
        """
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and cast to float32 for computation
        x = inputs.to(torch.float32).contiguous()
        B, S, N = x.shape
        rows = B * S
        x_2d = x.reshape(rows, N)

        # 1) Compute per-row mean and std (population, unbiased=False)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 256
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

        # 3) Apply gating via 2D Triton kernel over tiles of N
        out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

        # Reshape back and cast to bfloat16
        out = out_2d.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
