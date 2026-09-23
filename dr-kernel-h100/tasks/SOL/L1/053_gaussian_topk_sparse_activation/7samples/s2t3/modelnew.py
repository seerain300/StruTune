import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and std (population) along the last dimension,
# and also compute the scalar inv_norm_cdf(target_sparsity) using the Abramowitz & Stegun 5.2.23 approximation.
# Inputs:
#   x_ptr: pointer to input flattened as [rows, N], float32
#   mean_out_ptr: pointer to output mean per row, shape [rows], float32
#   std_out_ptr: pointer to output std per row, shape [rows], float32
#   N: number of columns (intermediate_size), int32 scalar
#   target_sparsity: float32 scalar in (0,1), passed as constexpr
# Output:
#   mean_out_ptr[row] = mean of row
#   std_out_ptr[row]  = std of row (population std)
#   The inv_norm_cdf is computed and stored in mean_out_ptr[rows] (we index out-of-bounds for rows)
# Note: This kernel performs two passes over the row: one to compute sum and sumsq, and another to compute mean and store std. We could
#       try to store both in one pass, but we need mean before we can write std; two passes is acceptable and simple.
@triton.jit
def reduce_mean_std_invndtri_kernel(
    x_ptr,                 # *const float32, size = rows * N
    mean_out_ptr,          # *float32, size = rows
    std_out_ptr,           # *float32, size = rows
    N: tl.constexpr,       # int32
    target_sparsity,       # float32 (scalar), constexpr
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    # Compute row base pointer: row starts at index row * N
    # First pass: accumulate sum and sum of squares
    sum_val = 0.0
    sumsq_val = 0.0
    cols = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + cols
        mask = idx < N
        offs = row * N + idx
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # Sum and sumsq in float32
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    mean = sum_val / N
    # population std: sqrt(E[x^2] - mean^2)
    var = sumsq_val / N - mean * mean
    # Ensure numerical stability: var >= 0 (due to rounding it might be slightly negative)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store mean and std for this row
    tl.store(mean_out_ptr + row, mean)
    tl.store(std_out_ptr + row, std)

    # Compute inv_norm_cdf(target_sparsity) in-kernel using Abramowitz & Stegun 5.2.23
    # The output is written at index rows (one past valid rows), not used further.
    # We compute per-kernel to avoid host overhead.
    # Note: We replicate the piecewise approximation.

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

    # Compute inv_norm_cdf(target_sparsity) into scalar slot at index rows
    # We'll write to mean_out_ptr[rows] (this is out-of-bounds for rows, but we can read it after in host)
    # But since Triton kernels cannot return scalars, we write to mean_out_ptr[rows] and read after.
    # However, Triton kernels do not support writing to out-of-bounds. So we instead write to mean_out_ptr[0] if rows >= 1,
    # or create a separate buffer. To keep it clean, we use a 1-element tensor passed in/out. Simpler: compute and store to a provided scalar buffer.

    # We'll store inv_cdf to std_out_ptr[rows] as a placeholder; it won't be used. Better: pass an out buffer.
    # Since we cannot store to a scalar out-of-kernel, we will compute and store into a per-row slot and then read after,
    # but we need a 1-element buffer. Triton doesn't support out-of-bounds write; so we instead compute and pass via an input-output parameter? Not supported.
    # Therefore, we will not compute inv_cdf in this kernel. We'll compute it in a separate small Triton kernel.

    # For now, just compute and store (we'll do it in a separate kernel below). This kernel will only compute mean and std.

    # Post: no need to return anything; we will launch a small Triton kernel to compute inv_cdf in ModelNew.forward.

    # Done.

# The above kernel only computes mean and std. We need a separate kernel to compute inv_norm_cdf(target_sparsity).
# We will do it as a tiny 1-element kernel to ensure Triton-only compute.

@triton.jit
def compute_inv_ndtri_kernel(out_ptr, target_sparsity, BLOCK_SIZE: tl.constexpr):
    # Single program writes inv_norm_cdf to out_ptr[0]
    # Implement Abramowitz & Stegun 5.2.23 approximation
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
        # lower region
        u = 1.0 - target_sparsity
        q = tl.sqrt(-2.0 * tl.log(u))
        inv_cdf = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    elif target_sparsity > p_high:
        # upper region
        u = target_sparsity
        q = tl.sqrt(-2.0 * tl.log(u))
        inv_cdf = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    else:
        # central region
        q = target_sparsity - 0.5
        r = q * q
        inv_cdf = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                  (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Store into out_ptr[0]
    tl.store(out_ptr, inv_cdf)


# Triton kernel: elementwise gating per row
# Inputs:
#   x_ptr: *const float32, flattened [rows, N]
#   mean_ptr: *const float32, shape [rows]
#   std_ptr: *const float32, shape [rows]
#   inv_cdf_ptr: *const float32, shape [1]
#   out_ptr: *float32, flattened [rows, N]
# Computes: out[row, :] = max(0, x[row, :] - (mean[row] + std[row] * inv_cdf[0]))
@triton.jit
def gate_rows_kernel(
    x_ptr,               # *const float32, size = rows * N
    mean_ptr,            # *const float32, size = rows
    std_ptr,             # *const float32, size = rows
    inv_cdf_ptr,         # *const float32, size = 1
    out_ptr,             # *float32, size = rows * N
    N: tl.constexpr,     # int32
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offs = row * N + cols
    mask = cols < N  # mask for partial last block

    # Load row data
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    threshold = mean + std * inv_cdf
    gated = x_vals - threshold
    gated = tl.maximum(gated, 0.0)

    # Store output
    tl.store(out_ptr + offs, gated, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle no sparsity
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and use float32 for compute
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Shape: [B, S, N]
        B, S, N = x_f32.shape
        rows = B * S

        # Allocate outputs for mean and std per row (float32)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Allocate input/output views flattened as [rows, N]
        x_flat = x_f32.view(rows, N)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # Compute per-row mean and std in Triton
        # Launch one program per row
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std_invndtri_kernel[grid](
            x_flat, mean, std, N, target_sparsity, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Compute inv_norm_cdf(target_sparsity) via tiny Triton kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_kernel[(1,)](inv_cdf_buf, target_sparsity, BLOCK_SIZE=1, num_warps=1)

        # Apply gating in Triton: y = max(0, x - (mean + std * inv_cdf))
        gate_rows_kernel[grid](
            x_flat, mean, std, inv_cdf_buf, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out