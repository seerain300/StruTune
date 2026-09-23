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
def gate_rows_2d_with_threshold(x_ptr, mean_ptr, std_ptr, out_ptr,
                                rows, N, inv_cdf, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating per row, using a per-row threshold:
      threshold = mean[row] + std[row] * inv_cdf(target_sparsity)
      y = max(0, x - threshold)
    x_ptr: input [rows, N] contiguous
    mean_ptr/std_ptr: per-row scalars [rows]
    out_ptr: output [rows, N] contiguous
    rows, N: int32
    inv_cdf: float32 scalar computed on host
    """
    row_id = tl.program_id(axis=0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    threshold = mean + std * inv_cdf

    # Process the row in tiles across columns
    num_tiles = tl.cdiv(N, BLOCK_SIZE)
    col_pid = tl.program_id(axis=1)
    start = col_pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    y = x - threshold  # broadcast scalar threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def _ndtri_approx(p: float) -> float:
    """
    Abramowitz and Stegun approximation for the standard normal inverse CDF (quantile).
    This is a 5th-order rational approximation and is accurate for p in (0, 1).
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p <= p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return result
    elif p >= p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return result
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        result = poly * q / denom
        return result


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel to compute inverse normal CDF via Abramowitz & Stegun approximation.
    out_ptr: 1-element tensor to store result as float32
    p: float32 scalar in (0, 1)
    """
    # Compute approximation and store
    inv_cdf = _ndtri_approx(p)
    tl.store(out_ptr, inv_cdf)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and std over last dim (population, unbiased=False).
        - Compute per-row threshold: mean + std * inv_cdf(target_sparsity).
        - Apply gating: y = max(0, x - threshold), return in bfloat16.
        All tensor ops are performed by Triton kernels; host code only orchestrates launches.
        """
        # Ensure float32 for numerical stability, keep original shape
        B, S, N = x.shape
        rows = B * S
        x_f32 = x.to(torch.float32)

        # Flatten to [rows, N] contiguous for kernels
        x_2d = x_f32.view(rows, N)

        # 1) Compute per-row mean and std via Triton
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

        # 3) Gating: apply threshold per row directly in Triton
        out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d_with_threshold[grid](x_2d, mean, std, out_2d, rows, N, inv_cdf_buf[0], BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back and cast to bfloat16
        out = out_2d.view(B, S, N).to(torch.bfloat16)
        return out