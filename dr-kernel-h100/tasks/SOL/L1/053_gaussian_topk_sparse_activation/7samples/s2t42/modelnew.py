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
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating kernel:
    For each row r, apply y = max(0, x_r - (mean_r + std_r * inv_cdf))
    x_ptr: [rows, N]
    mean_ptr/std_ptr: [rows]
    inv_cdf_ptr: [1] scalar float
    out_ptr: [rows, N]
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def _ndtri_approx(p: float) -> float:
    """
    Inverse of standard normal CDF using Abramowitz & Stegun 7.1.26 approximation.
    Numerically stable for p in (0, 1). Returns float.
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

    # For central region, use p directly; for tails, use sqrt(-log) transformation
    # We return a single scalar approximation; p is passed as float.
    q = 0.0
    if p < p_low:
        q = torch.sqrt(torch.tensor(-2.0 * torch.log(torch.tensor(p)), dtype=torch.float32)).item()
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return -poly / denom
    elif p > p_high:
        q = torch.sqrt(torch.tensor(-2.0 * torch.log(torch.tensor(1.0 - p)), dtype=torch.float32)).item()
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return poly / denom
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return poly * q / denom


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run.
    Computes mean and std per [B, S] row over last dim, computes inv_norm_cdf via PyTorch,
    and applies gating y = max(0, x - (mean + std * inv_cdf)) in Triton, output bfloat16.
    """
    # Expect inputs shaped [B, S, N]
    B, S, N = inputs.shape
    rows = B * S

    # Cast to float32 for computation
    x = inputs.contiguous().to(torch.float32)
    x_2d = x.view(rows, N)

    # 1) Compute per-row mean and std (population, unbiased=False)
    mean = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) using PyTorch (approx) to avoid Triton scalar issues
    inv_cdf = _ndtri_approx(float(target_sparsity))  # scalar float

    # 3) Apply gating using 2D Triton kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

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