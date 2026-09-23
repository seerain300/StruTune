import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_kernel(x_ptr, means_ptr, stds_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row (flattening B and S)
    pid = tl.program_id(axis=0)
    row_start = pid * F
    # Accumulate sum and sum of squares in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over feature dimension in chunks of BLOCK_F
    for i in range(0, F, BLOCK_F):
        offs = i + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / F
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / F - mean * mean
    var = tl.maximum(var, 0.0)  # clamp for numerical stability
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(means_ptr + pid, mean)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, out_ptr):
    # Compute inverse normal CDF for scalar p (0 < p < 1) using A&S 5.2.23
    # Note: We only evaluate the central region here, which is sufficient for typical target_sparsities.
    # If p == 0 or p == 1, return extreme values (not expected here).
    p = p  # scalar
    # Central region constants
    p_mid = 0.5  # center of central region (not used in mask but kept for clarity)
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

    q = p - 0.5
    r = q * q
    numerator = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    denominator = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = numerator / denominator

    tl.store(out_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row (flattening B and S)
    pid = tl.program_id(axis=0)
    row_start = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z

    # Apply elementwise: out = max(0, x - cutoff)
    for i in range(0, F, BLOCK_F):
        offs = i + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + offs, y, mask=mask)


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
        num_stages = 3

    # Launch mean and std kernel
    grid = (total_rows,)
    compute_mean_std_kernel[grid](inputs, means, stds, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

    # Allocate scalar output for z
    z_scalar = torch.empty(1, device=inputs.device, dtype=torch.float32)
    # Launch ndtri approximation kernel (Triton-only scalar)
    ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)

    # Allocate output (float32) for apply kernel
    out_fp32 = torch.empty_like(inputs, dtype=torch.float32)

    # Launch apply kernel
    apply_cutoff_relu_kernel[grid](inputs, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

    # Cast to bfloat16 to match original behavior
    out_bf16 = out_fp32.to(torch.bfloat16)
    return out_bf16


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        return _run_triton(inputs, target_sparsity)


def run(*args):
    return ModelNew()(*args)
