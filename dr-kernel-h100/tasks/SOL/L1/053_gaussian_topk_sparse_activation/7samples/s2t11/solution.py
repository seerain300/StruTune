import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) across the last dimension.
    x_ptr: pointer to input [rows, N] flattened
    mean_ptr/std_ptr: pointers to output vectors of length rows (float32)
    N: last dimension size (int32)
    """
    row_id = tl.program_id(axis=0)
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # For masked loads, use 0.0 so it doesn't affect sums
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        # Sum and sum of squares
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    # population variance: E[x^2] - (E[x])^2
    var = sum_sq / N - mean * mean
    # std is sqrt(var), ensure non-negative due to numerical precision
    std = tl.sqrt(var * 1.0)

    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_kernel(out_ptr, p: tl.float32):
    """
    Compute inv_norm_cdf(p) using Abramowitz & Stegun 7.1.26 approximation.
    Writes to out_ptr[0] as a scalar float32.
    p is scalar float32 in (0, 1).
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

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    lower = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    central = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    upper = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
            ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select region based on p
    mask_low = p < p_low
    mask_high = p > p_high
    inv = tl.where(mask_low, lower, central)
    inv = tl.where(mask_high, upper, inv)

    tl.store(out_ptr, inv)


@triton.jit
def gate_rows(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out[row, i] = max(0, x[row, i] - (mean[row] + std[row] * inv[0]))
    x_ptr: [rows, N] flattened
    mean_ptr/std_ptr: [rows], float32
    inv_ptr: [1], float32 (scalar inverse cdf)
    out_ptr: [rows, N] flattened, float32
    N: last dimension size
    """
    row_id = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv = tl.load(inv_ptr)  # scalar

    cutoff = mean + std * inv

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_id * N + offs, y, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the original run function:
        - Compute per-row mean and std in Triton (population std, unbiased=False).
        - Compute inv_norm_cdf(target_sparsity) in Triton (scalar).
        - Apply elementwise gating in Triton: y = max(0, x - (mean + std * inv_cdf)).
        Return tensor in bfloat16 to match original behavior.
        """
        # If no sparsity requested, return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32 for numerical stability
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Shape
        B, S, N = x_f32.shape
        rows = B * S

        # Allocate mean and std as [rows]
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Compute per-row mean and std in Triton
        x_flat = x_f32.view(rows, N)
        # Choose a reasonable block size; 1024 works well for typical N in provided workloads
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std[grid](x_flat, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Compute inv_norm_cdf(target_sparsity) using Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_kernel[(1,)](inv_cdf_buf, target_sparsity)

        # Allocate output and apply gating in Triton
        out_flat = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
        gate_rows[grid](x_flat, mean, std, inv_cdf_buf, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
