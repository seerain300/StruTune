import torch
import triton
import triton.language as tl

# Triton kernel: compute per-row mean and std across last dimension, and compute scalar inv_norm_cdf(target_sparsity).
# Inputs:
#   x_flat: [rows, N], float32
#   mean: [rows], float32
#   std: [rows], float32
#   N: int, last dimension size
#   target_sparsity: float
# Outputs:
#   mean[i], std[i] for i in [0, rows)
#   inv_cdf is computed inside the kernel and stored to mean[rows] (we will not use it; Triton doesn't support 2D return)
@triton.jit
def reduce_mean_std_invndtri_kernel(x_flat, mean, std, N, target_sparsity, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # row id in [0, rows)
    # Accumulators in float32
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over the last dimension in chunks
    for offs in range(0, N, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        row_ptr = x_flat + pid * N + cols
        x = tl.load(row_ptr, mask=mask, other=0.0)
        # Reduce across the BLOCK_SIZE vector
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and population std: var = E[x^2] - (E[x])^2
    n = N
    mean_val = sum_val / n
    var = sum_sq / n - mean_val * mean_val
    # Avoid tiny negative due to floating error
    var = tl.maximum(var, 0.0)
    std_val = tl.sqrt(var)

    # Store per-row stats
    tl.store(mean + pid, mean_val)
    tl.store(std + pid, std_val)

    # Compute inv_norm_cdf(target_sparsity) using Abramowitz & Stegun approximation (stored to mean[rows] as a placeholder).
    # Note: We won't read this back; we pass it as a host-side scalar to the gating kernel. This kernel just computes it.
    # We'll store to mean[rows] to use some memory; not used in forward.
    rows_total = tl.num_programs(axis=0)
    # Lower region constants
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

    # We need to compute inv_cdf for a single scalar target_sparsity; use pid==rows_total to store.
    if pid == rows_total:
        # Determine region
        if target_sparsity < p_low:
            q = tl.sqrt(-2.0 * tl.log(target_sparsity))
            inv_cdf = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                      (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        elif target_sparsity > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
            inv_cdf = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        else:
            q = target_sparsity - 0.5
            r = q * q
            inv_cdf = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

        # Store to mean[rows] as placeholder; not used in forward
        tl.store(mean + rows_total, inv_cdf)

# Triton kernel: per-row gating y = max(0, x - (mean + std * inv_cdf))
@triton.jit
def gate_rows_kernel(x_flat, mean, std, inv_cdf, out_flat, N, inv_scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Load per-row mean and std
    mean_val = tl.load(mean + pid)
    std_val = tl.load(std + pid)
    cutoff = mean_val + std_val * inv_scale  # inv_scale = inv_cdf * target_sparsity (target_sparsity passed as host scalar)
    # Iterate over columns in chunks and apply gating
    for offs in range(0, N, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        row_in_ptr = x_flat + pid * N + cols
        row_out_ptr = out_flat + pid * N + cols
        x = tl.load(row_in_ptr, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(row_out_ptr, y, mask=mask)

def _ndtri_scalar(target_sparsity: float) -> float:
    """Abramowitz and Stegun approximation for standard normal inverse CDF."""
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

    if target_sparsity < p_low:
        q = math.sqrt(-2.0 * math.log(target_sparsity))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    elif target_sparsity > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - target_sparsity))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = target_sparsity - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return input cast to bfloat16
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous
        x = inputs.contiguous()
        # Compute in float32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Flatten to [rows, N]
        B, S, N = x_f32.shape
        rows = B * S
        x_flat = x_f32.view(rows, N)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # Allocate per-row mean and std
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Compute inv_norm_cdf(target_sparsity) on host using A&S approximation
        inv_cdf_scalar = _ndtri_scalar(target_sparsity)

        # Choose BLOCK_SIZE for reduction; 1024 works well for typical sizes
        BLOCK_SIZE = 1024
        grid = (rows,)

        # Run reduction kernel to compute per-row mean and std
        reduce_mean_std_invndtri_kernel[grid](
            x_flat, mean, std, N, target_sparsity, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Elementwise gating: y = max(0, x - (mean + std * inv_cdf))
        # inv_scale = inv_cdf * target_sparsity
        inv_scale = inv_cdf_scalar * float(target_sparsity)

        # Run gating kernel
        gate_rows_kernel[grid](
            x_flat, mean, std, inv_scale, out_flat, N, inv_scale, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out