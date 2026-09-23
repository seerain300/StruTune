import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)
    # Iterate over features in chunks of BLOCK_F
    for i in range(0, F, BLOCK_F):
        idx = i + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sum_sq = tl.zeros((), dtype=tl.float32)
    # Iterate over features in chunks of BLOCK_F
    for i in range(0, F, BLOCK_F):
        idx = i + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    # Population std (unbiased=False)
    std = tl.sqrt(sum_sq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, out_ptr):
    # Abramowitz and Stegun 5.2.23 approximation for inverse normal CDF
    # Regions:
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Coefficients for lower region (x < p_low or x > p_high)
    c1 = -7.784894e-03
    c2 = -3.223965e-01
    c3 = -2.400758e+00
    c4 = -2.549733e+00
    c5 = 4.374664e+00
    c6 = 2.938164e+00

    d1 = 7.784696e-03
    d2 = 3.224671e-01
    d3 = 2.445134e+00
    d4 = 3.754409e+00

    # Coefficients for central region
    a1 = -3.969683e+01
    a2 = 2.209461e+02
    a3 = -2.759285e+02
    a4 = 1.383578e+02
    a5 = -3.066480e+01
    a6 = 2.506628e+00

    b1 = -5.447610e+01
    b2 = 1.615858e+02
    b3 = -1.556990e+02
    b4 = 6.680131e+01
    b5 = -1.328068e+01

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    num = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q2
    den = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    z_mid = num / den

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1*q3 + c2)*q3 + c3)*q3 + c4)*q3 + c5)*q3 + c6) / ((((d1*q3 + d2)*q3 + d3)*q3 + d4)*q3 + 1.0)

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_up = p > p_high
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_up, z_up, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar z (inverse normal CDF of target_sparsity)
    cutoff = mean + std * z
    # Apply elementwise y = max(0, x - cutoff)
    for i in range(0, F, BLOCK_F):
        idx = i + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_offset + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Allocate per-row vectors (float32) on device
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Heuristic tuning for BLOCK_F, num_warps, num_stages
        if F <= 4096:
            BLOCK_F = 4096
            num_warps = 4
            num_stages = 2
        elif F <= 8192:
            BLOCK_F = 8192
            num_warps = 8
            num_stages = 2
        else:
            BLOCK_F = 16384  # for F up to 16384, reduce loop iterations
            num_warps = 8
            num_stages = 2

        # Launch mean and std kernels
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # 1-element device tensor for z, filled by Triton ndtri kernel
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)  # pass scalar directly

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
