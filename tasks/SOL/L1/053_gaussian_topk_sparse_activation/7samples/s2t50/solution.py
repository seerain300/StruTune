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

    # Fast path: if N fits into one block, do a single vectorized load with correct row offset
    if N <= BLOCK_SIZE:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val = tl.sum(x)
        sum_sq = tl.sum(x * x)
    else:
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
    Triton scalar kernel computing inverse normal CDF (quantile function) for p in (0, 1).
    Uses Abramowitz & Stegun (5.2.23) approximation.
    Writes result to out_ptr[0] as float32.
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

    # Compute branches
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    t = c1 * q + c2
    t = t * q + c3
    t = t * q + c4
    t = t * q + c5
    numerator_l = t * q + c6
    t = d1 * q + d2
    t = t * q + d3
    t = t * q + d4
    denom_l = t * q + 1.0
    phi_l = numerator_l / denom_l

    # Central region
    q = p - 0.5
    r = q * q
    t = a1 * r + a2
    t = t * r + a3
    t = t * r + a4
    t = t * r + a5
    t = t * r + a6
    numerator_c = t * q
    t = b1 * r + b2
    t = t * r + b3
    t = t * r + b4
    t = t * r + b5
    denom_c = t * r + 1.0
    phi_c = numerator_c / denom_c

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    t = c1 * q + c2
    t = t * q + c3
    t = t * q + c4
    t = t * q + c5
    numerator_u = t * q + c6
    t = d1 * q + d2
    t = t * q + d3
    t = t * q + d4
    denom_u = t * q + 1.0
    phi_u = -numerator_u / denom_u

    # Select based on p
    mask_low = p < p_low
    mask_high = p > p_high
    phi = tl.where(mask_low, phi_l, 0.0)
    phi = tl.where(mask_high, phi_u, phi)
    # Default central region
    phi = tl.where(mask_low | mask_high, phi, phi_c)

    tl.store(out_ptr, phi)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating kernel: out[row, col] = max(0, x[row, col] - (mean[row] + std[row] * inv_cdf))
    x_ptr: input [rows, N], float32
    mean_ptr/std_ptr: per-row scalars [rows], float32
    inv_cdf_ptr: scalar [1], float32
    out_ptr: output [rows, N], float32
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE

    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row and per-row stats
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    # Compute gating
    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def _ndtri(p: float) -> float:
    # Helper for CPU fallback (if ever needed): Triton kernel is used in forward.
    # Implementing in Python would not be Triton-only; avoid here.
    raise RuntimeError("Use Triton kernel in forward instead of CPU _ndtri")


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
    1) Compute mean and std of input across feature dimension (last dim)
    2) Calculate threshold = mean + std * inv_norm_cdf(target_sparsity)
    3) Apply ReLU(input - threshold) to create sparse activations

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1]
    Returns:
        Sparsified tensor of same shape as input (dtype bfloat16).
    """
    # Ensure we are on CUDA; Triton requires CUDA tensors.
    if not inputs.is_cuda:
        inputs = inputs.cuda()

    # Compute in float32 for numerical stability
    x = inputs.contiguous()
    x_f32 = x.to(torch.float32)

    B, S, N = x_f32.shape
    rows = B * S

    # 1) Per-row mean and std via Triton reduction
    x_2d = x_f32.view(rows, N)
    mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

    # 3) Apply gating using 2D Triton kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

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
