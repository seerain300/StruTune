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
        offs = i + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs = x_ptr + row_start + offs
        x = tl.load(ptrs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * F
    mean = tl.load(means_ptr + pid)
    total = tl.zeros((), dtype=tl.float32)
    for i in range(0, F, BLOCK_F):
        offs = i + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs = x_ptr + row_start + offs
        x = tl.load(ptrs, mask=mask, other=0.0)
        diff = x - mean
        total += tl.sum(diff * diff, axis=0)
    var = total / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p: tl.float32, out_ptr):
    # Abramowitz & Stegun 5.2.23 piecewise approximation
    # Lower region
    p_low = 0.02425
    p_low_mask = p < p_low
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

    # For lower region, q = sqrt(-2 log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z = tl.where(p_low_mask, z_low, 0.0)

    # Central region
    p_high = 1.0 - p_low
    central_mask = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    numerator = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denominator = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = numerator / denominator
    z = tl.where(central_mask, z_mid, z)

    # Upper region
    upper_mask = p > p_high
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
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
        offs = i + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs = x_ptr + row_start + offs
        x = tl.load(ptrs, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        out_ptrs = out_ptr + row_start + offs
        tl.store(out_ptrs, y, mask=mask)


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
    if F < 2048:
        BLOCK_F = 2048
        num_warps = 4
        num_stages = 1
    elif F < 4096:
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

    # Allocate scalar z and compute via Triton kernel (no torch.tensor on host)
    z_scalar = torch.empty(1, device=inputs.device, dtype=torch.float32)
    ndtri_approx_kernel[(1,)](float(target_sparsity), z_scalar)

    # Allocate output (float32) for apply kernel
    out_fp32 = torch.empty_like(inputs, dtype=torch.float32)

    # Launch apply kernel
    apply_cutoff_relu_kernel[grid](inputs, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

    # Cast to bfloat16 to match original behavior
    return out_fp32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect inputs as [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise ValueError("ModelNew expects a single input tensor.")
        return _run_triton(*args)


def run(*args):
    return ModelNew()(*args)
