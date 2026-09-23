import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_lastdim_kernel(x_ptr, means_ptr, stds_ptr,
                             B, S, F,
                             BLOCK_F: tl.constexpr):
    # One program per row (row = batch*seq index)
    row = tl.program_id(0)
    b = row // S
    s = row % S
    row_offset = (b * S + s) * F

    acc = 0.0
    sumsq = 0.0

    # Single pass: accumulate sum and sum of squares across features
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        sumsq += tl.sum(vals * vals, axis=0)

    mean = acc / F
    # Population variance (unbiased=False): var = E[x^2] - (E[x])^2
    var = sumsq / F - mean * mean
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(means_ptr + row, mean)
    tl.store(stds_ptr + row, std)


@triton.jit
def ndtri_approx_kernel(p_ptr, z_ptr):
    # Compute inverse normal CDF for scalar p using Abramowitz & Stegun 5.2.23
    p_low = 0.02425
    p = tl.load(p_ptr)

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= 1.0 - p_low)
    mask_high = p > (1.0 - p_low)

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

    # Initialize z to 0.0
    z = 0.0

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    z = tl.where(p < p_low, z_low, z)

    # Central region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = tl.where((p >= p_low) & (p <= (1.0 - p_low)), z_mid, z)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    z = tl.where(p > (1.0 - p_low), z_high, z)

    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, means_ptr, stds_ptr, z_scalar_ptr, out_ptr,
                             B, S, F,
                             BLOCK_F: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    b = row // S
    s = row % S
    row_offset = (b * S + s) * F

    mean = tl.load(means_ptr + row)
    std = tl.load(stds_ptr + row)
    z = tl.load(z_scalar_ptr)
    cutoff = mean + std * z

    # Elementwise apply: y = max(x - cutoff, 0)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_offset + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Ensure CUDA and contiguity; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape

        # Allocate per-row means and stds (float32)
        means = torch.empty(B * S, device=x.device, dtype=torch.float32)
        stds = torch.empty(B * S, device=x.device, dtype=torch.float32)

        # Launch fused mean/std kernel
        grid = (B * S,)
        # Choose num_warps based on F to balance performance
        num_warps = 8 if F >= 8192 else 4
        mean_std_lastdim_kernel[grid](x, means, stds, B, S, F, BLOCK_F=4096, num_warps=num_warps)

        # Compute ndtri(target_sparsity) for scalar z
        p = torch.tensor(target_sparsity, device=x.device, dtype=torch.float32).reshape(1)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_approx_kernel[(1,)](p, z_scalar)

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, means, stds, z_scalar, out_fp32, B, S, F, BLOCK_F=4096, num_warps=num_warps)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)

        return out_bf16


def run(*args):
    return ModelNew()(*args)
