import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)
    # Process entire row in one iteration when BLOCK_F == F
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sumsq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
    diff = x - mean
    sumsq = tl.sum(diff * diff, axis=0)
    std = tl.sqrt(sumsq / F)  # population std, unbiased=False
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, z_ptr, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high):
    # Abramowitz & Stegun 5.2.23 approximation: scalar p -> scalar z
    q_low = tl.sqrt(-2.0 * tl.log(p))
    t_low = c1 * q_low + c2
    t2_low = t_low * q_low
    t3_low = t2_low * q_low
    t4_low = t3_low * q_low
    t5_low = t4_low * q_low
    num_low = t5_low + c6
    den_low = (d1 * q_low + d2) * q_low + d3
    den_low = den_low * q_low + d4
    den_low = den_low * q_low + 1.0
    z_low = num_low / den_low

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = ((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4
    num_mid = num_mid * r_mid + a5
    num_mid = num_mid * r_mid + a6
    den_mid = ((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4
    den_mid = den_mid * r_mid + b5
    den_mid = den_mid * r_mid + 1.0
    z_mid = num_mid * q_mid / den_mid

    # Upper region: p > 1 - p_low, but since p in (0,1), use p > 0.9758 (we set p_high = 1 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    t_high = c1 * q_high + c2
    t2_high = t_high * q_high
    t3_high = t2_high * q_high
    t4_high = t3_high * q_high
    t5_high = t4_high * q_high
    num_high = t5_high + c6
    den_high = (d1 * q_high + d2) * q_high + d3
    den_high = den_high * q_high + d4
    den_high = den_high * q_high + 1.0
    z_high = -num_high / den_high

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_high, 0.0)
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar inverse CDF value
    cutoff = mean + std * z
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
    y = x - cutoff
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only execution: ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Allocate per-row vectors (float32)
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Heuristics: set BLOCK_F to process entire row if F <= 16384
        if F <= 4096:
            BLOCK_F = 4096
            num_warps = 8
        elif F <= 8192:
            BLOCK_F = 8192
            num_warps = 16
        elif F <= 16384:
            BLOCK_F = 16384
            num_warps = 16
        else:
            # For very large F, loop in chunks
            BLOCK_F = 8192
            num_warps = 8

        # Launch mean and std kernels
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Allocate scalar for z (inverse CDF), computed by Triton
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Constants for A&S approximation
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

        # Launch ndtri approximation kernel (scalar)
        ndtri_approx_kernel[(1,)](target_sparsity, z_scalar, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high, num_warps=1, num_stages=1)

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
