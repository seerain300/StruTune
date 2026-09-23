import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, B, S, F, means_ptr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    total = tl.zeros((), dtype=tl.float32)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = vals - mean
        total += tl.sum(diff * diff, axis=0)
    var = total / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p_ptr, z_ptr):
    # p_ptr: [1] tensor with single float32 (target_sparsity)
    # z_ptr: [1] tensor to store result (inverse normal CDF)
    p = tl.load(p_ptr)
    p_low = 0.02425
    p_high = 1.0 - p_low

    # lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
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
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / denom_low

    # central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r = q_mid * q_mid
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
    poly_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid
    denom_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = poly_mid / denom_mid

    # upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / denom_high

    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, means_ptr, stds_ptr, z_ptr, out_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        out_vals = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + row_offset + idx, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous().to(torch.float32)
        B, S, F = inputs.shape

        # Per-row statistics (float32)
        means = torch.empty(B * S, device=inputs.device, dtype=torch.float32)
        stds = torch.empty(B * S, device=inputs.device, dtype=torch.float32)

        # Adaptive tuning for feature dimension
        if F <= 4096:
            BLOCK_F = 2048
            num_warps = 4
        elif F <= 8192:
            BLOCK_F = 4096
            num_warps = 8
        else:
            BLOCK_F = 8192
            num_warps = 8

        grid = (B * S,)

        # Launch mean kernel
        mean_lastdim_kernel[grid](inputs, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Launch std kernel
        std_lastdim_kernel[grid](inputs, stds, B, S, F, means, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Compute ndtri(target_sparsity) in Triton (scalar)
        p_tensor = torch.tensor(target_sparsity, device=inputs.device, dtype=torch.float32).reshape(1)
        z_scalar = torch.empty(1, device=inputs.device, dtype=torch.float32)
        ndtri_approx_kernel[(1,)](p_tensor, z_scalar)

        # Output buffer in float32
        out_fp32 = torch.empty_like(inputs, dtype=torch.float32)

        # Apply cutoff and ReLU
        apply_cutoff_relu_kernel[grid](inputs, means, stds, z_scalar, out_fp32, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Cast to bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
