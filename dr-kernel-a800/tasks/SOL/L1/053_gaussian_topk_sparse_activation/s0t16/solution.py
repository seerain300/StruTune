import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid
    row_offset = row * F
    total = 0.0
    for col in range(0, F, BLOCK_F):
        cols = col + tl.arange(0, BLOCK_F)
        mask = cols < F
        vals = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    mean = total / F
    tl.store(means_ptr + row, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid
    row_offset = row * F
    mean = tl.load(means_ptr + row)
    total = 0.0
    for col in range(0, F, BLOCK_F):
        cols = col + tl.arange(0, BLOCK_F)
        mask = cols < F
        vals = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        diff = vals - mean
        total += tl.sum(diff * diff, axis=0)
    var = total / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + row, std)


@triton.jit
def ndtri_approx_kernel(p_ptr, z_ptr):
    # p_ptr: 1-element tensor, float32, device
    # z_ptr: 1-element tensor, float32, device
    p = tl.load(p_ptr)  # scalar
    # Coefficients for Abramowitz & Stegun 5.2.23
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

    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / denom
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = -poly / denom
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly * q / denom

    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid
    row_offset = row * F

    mean = tl.load(means_ptr + row)
    std = tl.load(stds_ptr + row)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z

    for col in range(0, F, BLOCK_F):
        cols = col + tl.arange(0, BLOCK_F)
        mask = cols < F
        vals = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        out_vals = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + row_offset + cols, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguity
        if not x.is_cuda:
            x = x.to('cuda')
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape

        # Allocate per-row means and stds
        means = torch.empty(B * S, device=x.device, dtype=torch.float32)
        stds = torch.empty(B * S, device=x.device, dtype=torch.float32)

        # Launch reduction kernels: one program per row
        grid = (B * S,)
        num_warps = 8 if F >= 8192 else 4
        BLOCK_F = 4096

        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Compute z = ndtri(target_sparsity) in Triton using a 1-element device tensor
        p_tensor = torch.empty(1, device=x.device, dtype=torch.float32)
        # Write the target sparsity into the device tensor (no torch math on host)
        p_tensor[0] = float(target_sparsity)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_approx_kernel[(1,)](p_tensor, z_scalar)

        # Allocate output (float32) and apply cutoff + ReLU
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
