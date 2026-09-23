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
def compute_inv_ndtri_scalar(out_ptr, p_ptr):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) for p using A&S 26.2.23 approximation.
    p_ptr: 1-element tensor with target sparsity in [0,1]
    out_ptr: 1-element tensor to store inv_norm_cdf(p)
    """
    # Load p
    p = tl.load(p_ptr)  # float32
    # Constants for approximation
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
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        denom = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        result = -poly / denom
    # Central region
    elif p <= p_high:
        z = p - 0.5
        r = z * z
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * z
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        result = poly / denom
    # Upper region
    else:
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        denom = ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        result = poly / denom

    tl.store(out_ptr, result)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating: y = max(0, x - (mean + std * inv_cdf))
    Operates on tiles of columns per row. Grid = (rows, num_tiles).
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation:
    - Compute per-row mean and std (population) across last dim.
    - Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel.
    - Apply gating via Triton 2D kernel.
    Returns bfloat16 tensor with same shape as inputs.
    """
    # No torch operations on tensors in forward
    # Flatten to [rows, N] where rows = B * S, N = last dim
    B, S, N = inputs.shape
    rows = B * S

    # Ensure contiguous and compute in float32
    x = inputs.contiguous()
    x_f32 = x.view(rows, N).to(torch.float32)

    # 1) Per-row mean and std (population)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
    p_tensor = torch.tensor([target_sparsity], device=x.device, dtype=torch.float32)
    inv_cdf = torch.empty(1, device=x.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf, p_tensor)

    # 3) Elementwise gating using 2D Triton kernel
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_f32, mean, std, inv_cdf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape [batch_size, seq_len, intermediate_size]
        assert len(args) == 1, "ModelNew expects a single input tensor"
        return run(*args)


def run(*args):
    return ModelNew()(*args)
