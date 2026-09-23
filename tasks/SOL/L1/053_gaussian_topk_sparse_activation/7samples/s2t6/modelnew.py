import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and std (unbiased=False), and the scalar inv_norm_cdf(target_sparsity)
# Input:
#   x_flat: [rows, N] float32
#   mean_out: [rows] float32 (one element per row, will be written by this kernel)
#   std_out: [rows] float32 (one element per row, will be written by this kernel)
#   N: number of columns per row (int32)
#   target_sparsity: Python float scalar passed at launch
@triton.jit
def reduce_mean_std_invndtri_kernel(x_ptr, mean_out_ptr, std_out_ptr, N, target_sparsity,
                                    BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Base pointer for this row
    base = row_id * N
    # Accumulate sum and sum of squares in float32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over the last dimension in chunks
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        # Upcast to float32 for stable accumulation
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and population std (unbiased=False)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    # Ensure non-negative variance due to numeric issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute inv_norm_cdf(target_sparsity) using A&S 26.2.23 approximation (single scalar)
    # Note: The following constants match the original code.
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

    inv = 0.0
    if target_sparsity < p_low:
        q = tl.sqrt(-2.0 * tl.log(target_sparsity))
        inv = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif target_sparsity > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
        inv = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = target_sparsity - 0.5
        r = q * q
        inv = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Write results: per-row mean and std; scalar inv is computed but not stored here (forward passes it)
    tl.store(mean_out_ptr + row_id, mean)
    tl.store(std_out_ptr + row_id, std)


# Triton kernel: apply gating per row: y = max(0, x - (mean + std * inv_scale))
# Inputs:
#   x_flat: [rows, N] float32
#   mean: [rows] float32
#   std: [rows] float32
#   inv_scale: Python float passed at launch (value of inv_norm_cdf(target_sparsity) * target_sparsity)
# Outputs:
#   out_flat: [rows, N] float32
@triton.jit
def gate_rows_kernel(x_ptr, mean_ptr, std_ptr, inv_scale, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * N
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    # Compute threshold
    threshold = mean + std * inv_scale

    # Apply gating
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        # Gate: y = max(0, x - threshold)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return inputs unchanged in bfloat16
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and compute in float32
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Flatten to [rows, N]
        B, S, N = x_f32.shape
        rows = B * S
        x_flat = x_f32.view(rows, N)

        # Allocate per-row statistics
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Compute inv_norm_cdf(target_sparsity) using A&S approximation in Triton (scalar)
        # We pass target_sparsity as a Python float to the kernel and compute the scalar there.
        # To get the scalar for gating, we run a tiny kernel that produces a 1-element tensor containing the result.
        # However, since it's a scalar, we can compute it once and pass it to the gating kernel as a Python float.
        # Here, we compute it directly in Python using the same formula (to avoid another Triton launch).
        # But to keep Triton-only, we can instead compute it inside a small kernel and retrieve it via a view.
        # Simpler approach: compute it in Python using the same coefficients, then pass as float to gating kernel.
        # This keeps Triton-only for the heavy work. We'll do this to satisfy constraints cleanly.

        # Compute inv_norm_cdf using the same formula in Python (A&S 26.2.23)
        # This is just a host-side scalar and does not affect Triton-only requirement.
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

        # Compute inv_norm_cdf(target_sparsity) in Python
        if target_sparsity < p_low:
            q = (-(2.0 * math.log(target_sparsity))) ** 0.5
            inv = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        elif target_sparsity > p_high:
            q = (-(2.0 * math.log(1.0 - target_sparsity))) ** 0.5
            inv = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        else:
            q = target_sparsity - 0.5
            r = q * q
            inv = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                  (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

        inv_cdf_scalar = inv

        # Launch reduction kernel: one program per row
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std_invndtri_kernel[grid](x_flat, mean, std, N, target_sparsity, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Allocate output flat tensor
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # Scale for gating: threshold = mean + std * inv_cdf(target_sparsity) == x - std * inv_cdf is subtracted, here we use + std * inv_cdf
        # inv_scale = inv_cdf_scalar * target_sparsity
        inv_scale = inv_cdf_scalar * target_sparsity

        # Launch gating kernel
        gate_rows_kernel[grid](x_flat, mean, std, inv_scale, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out