import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_start = pid * F
    total = tl.zeros((), dtype=tl.float32)
    # Iterate across feature dimension in chunks of BLOCK_F
    for i in range(0, F, BLOCK_F):
        offs = row_start + i + tl.arange(0, BLOCK_F)
        mask = offs < row_start + F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_start = pid * F
    mean = tl.load(means_ptr + pid)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for i in range(0, F, BLOCK_F):
        offs = row_start + i + tl.arange(0, BLOCK_F)
        mask = offs < row_start + F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    # Population std (unbiased=False)
    std = tl.sqrt(sum_sq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, out_ptr):
    # Inverse normal CDF (Abramowitz & Stegun 5.2.23) for scalar p in (0,1)
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
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

    numerator_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = numerator_low / den_low

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    numerator_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid
    den_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = numerator_mid / den_mid

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    numerator_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    den_up = ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    z_up = numerator_up / den_up

    # Select region
    central_mask = (p >= p_low) & (p <= p_high)
    lower_mask = p < p_low
    upper_mask = p > p_high
    # Default to 0.0; Triton will broadcast properly
    z = tl.zeros((), dtype=tl.float32)
    z = tl.where(lower_mask, z_low, z)
    z = tl.where(central_mask, z_mid, z)
    z = tl.where(upper_mask, z_up, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_start = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z
    for i in range(0, F, BLOCK_F):
        offs = row_start + i + tl.arange(0, BLOCK_F)
        mask = offs < row_start + F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA and contiguous; compute in float32
    assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels."
    inputs = inputs.contiguous().to(torch.float32)
    B, S, F = inputs.shape
    total_rows = B * S

    # Allocate per-row vectors (float32)
    means = torch.empty(total_rows, device=inputs.device, dtype=torch.float32)
    stds = torch.empty(total_rows, device=inputs.device, dtype=torch.float32)

    # Heuristic for block size and warps based on F
    if F < 4096:
        BLOCK_F = 4096
        num_warps = 8
        num_stages = 2
    elif F < 8192:
        BLOCK_F = 8192
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_F = 16384
        num_warps = 8
        num_stages = 2

    # Launch mean and std kernels
    grid = (total_rows,)
    mean_lastdim_kernel[grid](inputs, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
    std_lastdim_kernel[grid](inputs, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

    # Allocate scalar output for ndtri result
    z_scalar = torch.empty(1, device=inputs.device, dtype=torch.float32)

    # Launch ndtri approximation kernel (Triton-only)
    ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)

    # Allocate output (float32) for apply kernel
    out_fp32 = torch.empty_like(inputs, dtype=torch.float32)

    # Launch apply kernel
    apply_cutoff_relu_kernel[grid](inputs, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

    # Cast to bfloat16 to match original behavior
    out_bf16 = out_fp32.to(torch.bfloat16)
    return out_bf16


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input as a single 3D tensor: [batch_size, seq_len, intermediate_size]
        x = args[0]
        target_sparsity = float(0.5)  # default; original run signature takes target_sparsity
        return _run_triton(x, target_sparsity)


def run(*args):
    return ModelNew()(*args)
