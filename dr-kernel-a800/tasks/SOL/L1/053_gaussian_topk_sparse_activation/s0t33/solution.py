import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = 0.0
    # Loop over features in chunks of BLOCK_F
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
    sumsq = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = x - mean
        sumsq += tl.sum(diff * diff, axis=0)
    # Population std: unbiased=False
    std = tl.sqrt(sumsq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(out_ptr, p, a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low, p_high):
    # Single scalar computation: inverse normal CDF for p in (0,1)
    # Piecewise approximation (Abramowitz & Stegun 5.2.23)
    q = 0.0
    # Lower region
    if p < p_low:
        u = -tl.log(p)
        q = tl.sqrt(2.0 * u)
        t = c1 * q + c2
        t = t * q + c3
        t = t * q + c4
        t = t * q + c5
        t = t * q + c6
        v = d1 * q + d2
        v = v * q + d3
        v = v * q + d4
        z = t / (v + 1.0)
    # Central region
    elif p <= p_high:
        u = p - 0.5
        r = u * u
        t = a1 * r + a2
        t = t * r + a3
        t = t * r + a4
        t = t * r + a5
        t = t * r + a6
        v = b1 * r + b2
        v = v * r + b3
        v = v * r + b4
        v = v * r + b5
        z = t / (v + 1.0)
    # Upper region
    else:
        u = -tl.log(1.0 - p)
        q = tl.sqrt(2.0 * u)
        t = c1 * q + c2
        t = t * q + c3
        t = t * q + c4
        t = t * q + c5
        t = t * q + c6
        v = d1 * q + d2
        v = v * q + d3
        v = v * q + d4
        z = -t / (v + 1.0)
    tl.store(out_ptr, z)


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
        # ReLU
        y = tl.maximum(y, 0.0)
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

        # Device scalar for z (inverse-normal CDF)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Triton-only ndtri approximation kernel (no torch.tensor creation on host)
        p_low = 0.02425
        p_high = 1.0 - p_low
        ndtri_approx_kernel[(1,)](
            z_scalar, float(target_sparsity),
            -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
            -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01,
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
            -3.223964580411365e-01, 2.445134137142996e+00, 3.754408661907416e+00, 7.784695709041462e-03,
            p_low, p_high
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
