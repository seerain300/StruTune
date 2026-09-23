import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input tensor viewed as [rows, N], contiguous along the last dim.
    mean_ptr/std_ptr: pointers to output vectors of length rows (float32)
    N: int32, size of last dimension
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
def compute_inv_ndtri_scalar(inv_ptr, p):
    """
    Compute inverse normal CDF (quantile) for a scalar p in (0, 1).
    Writes result into inv_ptr[0].
    Uses Abramowitz & Stegun approximation (formula 26.2.23).
    """
    # Constants
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

    # Piecewise approximation
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        y = poly / den
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        y = poly * q / den
    else:
        # Upper tail
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        y = -poly / den

    # Write result
    tl.store(inv_ptr, y)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: y = max(0, x - (mean + std * inv_cdf)).
    Grid is 2D: axis 0 = row index, axis 1 = tile index over N.
    x_ptr: [rows, N], float32
    mean_ptr/std_ptr: [rows], float32
    inv_ptr: [1], float32 (scalar inv_cdf)
    out_ptr: [rows, N], float32
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE

    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load inputs
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv = tl.load(inv_ptr)  # scalar

    # Compute gating
    thr = mean + std * inv
    y = x - thr
    y = tl.maximum(y, 0.0)

    # Store results
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation of the Gaussian-based top-k sparse activation.
        Computes per-row threshold mean + std * inv_norm_cdf(target_sparsity)
        and returns y = max(0, x - threshold).
        Output dtype matches original behavior (bfloat16).
        """
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        # Cast to float32 for computation
        x_f32 = x.contiguous().to(torch.float32)

        # Reshape to [rows, N] where rows = B*S and N = intermediate_size
        B, S, N = x_f32.shape
        rows = B * S
        x_2d = x_f32.view(rows, N)

        # 1) Compute per-row mean and std in Triton
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Use a chunk size that balances performance and robustness; loop covers any N
        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

        # 3) Apply gating via Triton 2D kernel over tiles of N
        out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back to [B, S, N] and cast to bfloat16
        out = out_2d.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
