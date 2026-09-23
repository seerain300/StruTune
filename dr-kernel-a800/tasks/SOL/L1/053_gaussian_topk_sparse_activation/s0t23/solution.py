import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)
    # Iterate over feature dimension in chunks
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sumsq = tl.zeros((), dtype=tl.float32)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = vals - mean
        sumsq += tl.sum(diff * diff, axis=0)
    # population std: divide by F
    std = tl.sqrt(sumsq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p_scalar, z_ptr):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF
    # Piecewise: lower, central, upper
    p = p_scalar  # scalar float32

    # Constants for lower region approximation
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

    # Lower region: p < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region: p_low <= p <= p_high
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region: p > p_high
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select region
    mask_low = p < p_low
    mask_up = p > p_high
    # Triton will handle scalar branching efficiently; we build the result via masks
    z = tl.where(mask_low, z_low, tl.where(mask_up, z_up, z_mid))

    # Store scalar z
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F

    # Load mean and std for this row
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar

    cutoff = mean + std * z
    # Loop over features in chunks
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        # Compute output = max(0, x - cutoff)
        out_vals = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + row_offset + idx, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Allocate per-row vectors (float32)
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Heuristic tuning for block size and warps
        if F < 4096:
            BLOCK_F = 4096
            num_warps = 8
        elif F < 8192:
            BLOCK_F = 8192
            num_warps = 8
        else:
            BLOCK_F = 16384
            num_warps = 8

        grid = (total_rows,)

        # Launch mean and std kernels
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Launch ndtri approximation kernel for scalar z
        ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
