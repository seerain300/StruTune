import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, NUM_CHUNKS: tl.constexpr, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)

    # Iterate over feature chunks
    k = 0
    while k < NUM_CHUNKS:
        offsets = k * BLOCK_F + tl.arange(0, BLOCK_F)
        mask = offsets < F
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
        k += 1

    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, NUM_CHUNKS: tl.constexpr, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sumsq = tl.zeros((), dtype=tl.float32)

    # Iterate over feature chunks
    k = 0
    while k < NUM_CHUNKS:
        offsets = k * BLOCK_F + tl.arange(0, BLOCK_F)
        mask = offsets < F
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        diff = vals - mean
        sumsq += tl.sum(diff * diff, axis=0)
        k += 1

    # Population std: unbiased=False, divide by F
    std = tl.sqrt(sumsq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, out_ptr):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF.
    # p is scalar in (0, 1); out_ptr points to a single float32 element.
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
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        z = num / den
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        z = -num / den
    else:
        # Central region
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = num / den

    tl.store(out_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, NUM_CHUNKS: tl.constexpr, BLOCK_F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar

    # Compute threshold per element: mean + std * z
    threshold = mean + std * z

    # Apply elementwise: out = max(0, x - threshold)
    k = 0
    while k < NUM_CHUNKS:
        offsets = k * BLOCK_F + tl.arange(0, BLOCK_F)
        mask = offsets < F
        vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        out_vals = vals - threshold
        out_vals = tl.maximum(out_vals, 0.0)
        tl.store(out_ptr + row_offset + offsets, out_vals, mask=mask)
        k += 1


def _choose_launch_params(F: int):
    # Choose BLOCK_F and num_warps to minimize NUM_CHUNKS while keeping good occupancy
    if F < 4096:
        return 4096, 8, 2
    elif F < 8192:
        return 8192, 8, 2
    else:
        return 16384, 8, 2


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Output buffer (float32)
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Allocate per-row vectors
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Launch params
        BLOCK_F, num_warps, num_stages = _choose_launch_params(F)
        NUM_CHUNKS = (F + BLOCK_F - 1) // BLOCK_F

        # Launch mean and std kernels (one program per row)
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, NUM_CHUNKS=NUM_CHUNKS, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, NUM_CHUNKS=NUM_CHUNKS, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # 1-element device tensor for z, filled by Triton ndtri kernel
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, NUM_CHUNKS=NUM_CHUNKS, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
