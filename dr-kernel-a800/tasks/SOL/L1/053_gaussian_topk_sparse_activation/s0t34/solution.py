import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = 0.0
    # Iterate over features in chunks of BLOCK_F
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    # Compute population variance: sum((x - mean)^2) / F
    var = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p_ptr, z_ptr,  # p_ptr points to a 1-element tensor with target_sparsity
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low, p_high):
    # Single program: compute z = inv-phi(p)
    # p is read from p_ptr[0]; write result to z_ptr[0]
    p = tl.load(p_ptr)
    # Promote constants to float32
    a1 = tl.full((), a1, tl.float32); a2 = tl.full((), a2, tl.float32); a3 = tl.full((), a3, tl.float32)
    a4 = tl.full((), a4, tl.float32); a5 = tl.full((), a5, tl.float32); a6 = tl.full((), a6, tl.float32)
    b1 = tl.full((), b1, tl.float32); b2 = tl.full((), b2, tl.float32); b3 = tl.full((), b3, tl.float32)
    b4 = tl.full((), b4, tl.float32); b5 = tl.full((), b5, tl.float32)
    c1 = tl.full((), c1, tl.float32); c2 = tl.full((), c2, tl.float32); c3 = tl.full((), c3, tl.float32)
    c4 = tl.full((), c4, tl.float32); c5 = tl.full((), c5, tl.float32); c6 = tl.full((), c6, tl.float32)
    d1 = tl.full((), d1, tl.float32); d2 = tl.full((), d2, tl.float32); d3 = tl.full((), d3, tl.float32)
    d4 = tl.full((), d4, tl.float32)
    # p_low, p_high as float32
    p_low_f = tl.full((), p_low, tl.float32)
    p_high_f = tl.full((), 1.0 - p_low_f, tl.float32)

    # Promote p to float32
    p = tl.full((), p, tl.float32)

    # Piecewise approximation
    # Lower region
    if p < p_low_f:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        tl.store(z_ptr, z)
        return
    # Central region
    if p <= p_high_f:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        tl.store(z_ptr, z)
        return
    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_offset + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Allocate per-row vectors
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Heuristics for block size and warps to minimize loop iterations
        if F <= 4096:
            BLOCK_F = 4096
            num_warps = 8
            num_stages = 2
        elif F <= 8192:
            BLOCK_F = 8192
            num_warps = 8
            num_stages = 2
        else:
            BLOCK_F = 16384
            num_warps = 8
            num_stages = 2

        # Launch mean and std kernels: one program per row
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Device scalar for z (inverse-normal CDF), Triton-only computation
        p_scalar = torch.tensor([float(target_sparsity)], device=x.device, dtype=torch.float32)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Constants for Abramowitz & Stegun 5.2.23
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00
        p_low = 0.02425

        # Launch Triton ndtri kernel
        ndtri_approx_kernel[(1,)](
            p_scalar, z_scalar,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low
        )

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
