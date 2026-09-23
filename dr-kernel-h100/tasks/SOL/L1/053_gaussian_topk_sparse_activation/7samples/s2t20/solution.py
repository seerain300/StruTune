import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input [rows, N] flattened (row-major)
    mean_ptr/std_ptr: pointers to per-row scalars [rows]
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
def compute_inv_ndtri_scalar(inv_ptr, p):
    """
    Compute inverse normal CDF (quantile) for a scalar p in (0,1).
    Uses Abramowitz & Stegun approximation (formula 26.2.23).
    Writes result to inv_ptr[0] (float32).
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

    # Select region
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        out = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        out = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        out = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(inv_ptr, out)


@triton.jit
def gate_row_per_row(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating per row: y = max(0, x - (mean + std * inv_cdf))
    x_ptr: [rows, N] input
    mean_ptr/std_ptr: [rows]
    inv_ptr: [1] scalar inv_cdf
    out_ptr: [rows, N] output
    """
    row_id = tl.program_id(axis=0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar

    # Compute threshold vector for this row
    thr = mean + std * inv_cdf

    # Process the row in one chunk of size BLOCK_SIZE (set to N at launch)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    y = x - thr  # broadcast scalar threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes:
          - per-row mean and std across last dim (population, unbiased=False)
          - inv_norm_cdf(target_sparsity) via A&S approximation
          - y = max(0, x - (mean + std * inv_cdf))
        Returns y in bfloat16.
        """
        assert input.is_cuda, "Input must be on CUDA device for Triton kernels"
        # Cast to float32 for numerical stability
        x = input.to(torch.float32)

        # Flatten to [rows, N] where N is the last dimension
        N = x.shape[-1]
        rows = x.numel() // N
        x_2d = x.view(rows, N)

        # 1) Compute per-row mean and std
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity))

        # 3) Apply gating per row in a single pass
        out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        # Set BLOCK_SIZE to N so each program handles a full row
        BLOCK_SIZE_GT = N  # specialize per N; Triton compiles per specialization
        gate_row_per_row[(rows,)](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back and cast to bfloat16 to match original behavior
        out = out_2d.view(*x.shape).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
