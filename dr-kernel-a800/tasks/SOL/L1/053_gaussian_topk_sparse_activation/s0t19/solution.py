import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row (pid in [0, B*S))
    pid = tl.program_id(axis=0)
    row = pid
    if row >= B * S:
        return
    base = row * F
    total_sum = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        total_sum += tl.sum(x, axis=0)
    mean = total_sum / F
    tl.store(means_ptr + row, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid
    if row >= B * S:
        return
    base = row * F
    mean = tl.load(means_ptr + row)
    total_var = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        diff = x - mean
        total_var += tl.sum(diff * diff, axis=0)
    var = total_var / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + row, std)


@triton.jit
def ndtri_approx_kernel(p_ptr, z_ptr):
    # Compute inverse normal CDF for scalar p (0 < p < 1) using A&S 5.2.23 approximation.
    p = tl.load(p_ptr)  # scalar float32

    # Piecewise constants
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
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

    poly_lower = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    denom_lower = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    z_lower = poly_lower / denom_lower

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    a6 = 2.506628277459239e+00
    poly_central = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q2
    denom_central = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_central = poly_central / denom_central

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_upper = (((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6)
    denom_upper = (((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0))
    z_upper = - (poly_upper / denom_upper)

    # Select piecewise
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    z = tl.where(mask_low, z_lower, 0.0)
    z = tl.where(mask_mid, z_central, z)
    z = tl.where(mask_high, z_upper, z)

    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, means_ptr, stds_ptr, z_ptr, out_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row = pid
    if row >= B * S:
        return
    base_x = row * F
    mean = tl.load(means_ptr + row)
    std = tl.load(stds_ptr + row)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z

    # Apply ReLU with adaptive threshold across features
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0)
        # Elementwise: max(0, x - cutoff). Note cutoff is scalar per row.
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base_x + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure on CUDA and contiguous
        assert inputs.is_cuda, "Input must be a CUDA tensor"
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape

        # Allocate per-row means and stds (float32)
        means = torch.empty(B * S, device=x.device, dtype=torch.float32)
        stds = torch.empty(B * S, device=x.device, dtype=torch.float32)

        # Choose BLOCK_F based on F
        if F >= 16384:
            BLOCK_F = 8192
            num_warps = 8
        else:
            BLOCK_F = 4096
            num_warps = 4

        # Launch mean and std kernels
        grid = (B * S,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Allocate 1-element tensor for ndtri result on device
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Launch ndtri approximation kernel
        p_tensor = torch.empty(1, device=x.device, dtype=torch.float32)
        # Write target_sparsity to device (purely for computation inside kernel; no host-side torch ops beyond allocation)
        # We can fill p_tensor via a dummy kernel, but Triton allows passing scalars via tensors; here we rely on calling the kernel with p_tensor.
        # Note: In Triton, we pass tensors for reading; writing happens inside the kernel. We initialize p_tensor to target_sparsity.
        # However, Triton cannot directly write to p_tensor; so we pass a read-only tensor and let the kernel compute z into z_scalar.
        # To ensure p_tensor is filled, we can perform a trivial load; but better is to construct it before launch. In practice, we pass a tensor containing the value.
        # Create p_tensor with the scalar target_sparsity
        p_tensor[0] = target_sparsity
        ndtri_approx_kernel[(1,)](p_tensor, z_scalar)

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply cutoff + ReLU kernel
        apply_cutoff_relu_kernel[grid](x, means, stds, z_scalar, out_fp32, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
