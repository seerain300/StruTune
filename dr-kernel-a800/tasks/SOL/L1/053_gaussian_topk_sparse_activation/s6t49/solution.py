import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(x_ptr, out_sum_ptr,
                     B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                     stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
                     BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    total = 0.0
    # Iterate over feature dimension in chunks of BLOCK_F
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        # Pointer to [B, S, F] contiguous layout: offset = row_b*S*F + row_s*F + offs
        ptr = x_ptr + row_b * S * F + row_s * F + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
    tl.atomic_add(out_sum_ptr + pid, total)


@triton.jit
def sumsq_rows_kernel(x_ptr, out_sumsq_ptr,
                      B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                      stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    total = 0.0
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptr = x_ptr + row_b * S * F + row_s * F + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.atomic_add(out_sumsq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr,
                         B: tl.constexpr, S: tl.constexpr, F: tl.constexpr):
    # One program per row
    pid = tl.program_id(axis=0)
    total = tl.load(out_sum_ptr + pid)
    total2 = tl.load(out_sumsq_ptr + pid)
    mean = total / F
    var = total2 / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(z_ptr,
                        p,  # scalar target_sparsity (Python float passed to kernel)
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low):
    # Compute inverse-normal CDF for p using Abramowitz & Stegun 7.1.26 approximation
    # Piecewise approximation with low/mid/high regions
    # Allocate output
    eps = 1e-7
    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    # Horner evaluation of numerator and denominator
    # numerator = c1*q + c2; for each subsequent: numerator = numerator*q + next
    numerator_low = c1
    numerator_low = numerator_low * q_low + c2
    numerator_low = numerator_low * q_low + c3
    numerator_low = numerator_low * q_low + c4
    numerator_low = numerator_low * q_low + c5
    numerator_low = numerator_low * q_low + c6

    den_low = d1
    den_low = den_low * q_low + d2
    den_low = den_low * q_low + d3
    den_low = den_low * q_low + d4

    result_low = numerator_low / (den_low + 1.0)

    # Mid region
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    u = p - 0.5
    r = u * u
    num_mid = a1
    num_mid = num_mid * r + a2
    num_mid = num_mid * r + a3
    num_mid = num_mid * r + a4
    num_mid = num_mid * r + a5
    num_mid = num_mid * r + a6
    num_mid = num_mid * u

    den_mid = b1
    den_mid = den_mid * r + b2
    den_mid = den_mid * r + b3
    den_mid = den_mid * r + b4
    den_mid = den_mid * r + b5
    den_mid = den_mid * r + 1.0

    result_mid = num_mid / den_mid

    # Upper region
    mask_high = (p > (1.0 - p_low)) & (p < (1.0 - eps))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    numerator_high = c1
    numerator_high = numerator_high * q_high + c2
    numerator_high = numerator_high * q_high + c3
    numerator_high = numerator_high * q_high + c4
    numerator_high = numerator_high * q_high + c5
    numerator_high = numerator_high * q_high + c6

    den_high = d1
    den_high = den_high * q_high + d2
    den_high = den_high * q_high + d3
    den_high = den_high * q_high + d4

    result_high = -numerator_high / (den_high + 1.0)

    # Combine using masks; result = mask_low * result_low + mask_mid * result_mid + mask_high * result_high
    # Triton doesn't support dynamic branching on tensor values; we use masks (boolean) and multiply
    # Cast masks to float for multiplication
    mask_low_f = mask_low.to(tl.float32)
    mask_mid_f = mask_mid.to(tl.float32)
    mask_high_f = mask_high.to(tl.float32)

    result = result_low * mask_low_f + result_mid * mask_mid_f + result_high * mask_high_f

    # Store to z_ptr[0]
    tl.store(z_ptr, result)


@triton.jit
def apply_threshold_kernel(x_ptr, mean_ptr, std_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
                           BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(out_ptr)  # scalar z computed by ndtri_scalar_kernel
    threshold = mean + std * z

    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptr = x_ptr + row_b * S * F + row_s * F + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + pid * F + (f_start + tl.arange(0, BLOCK_F)), y.to(tl.float32), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous layout for [B, S, F]
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Prepare accumulators for sums (float32)
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Choose BLOCK_F based on F for good throughput
        BLOCK_F = 1024 if F >= 8192 else (512 if F >= 2048 else 256)

        # Launch sum and sumsq kernels: one program per row
        grid = (B * S,)
        sum_rows_kernel[grid](
            inputs, out_sum,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=4, num_stages=2
        )
        sumsq_rows_kernel[grid](
            inputs, out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=4, num_stages=2
        )

        # Compute per-row mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](
            out_sum, out_sumsq, out_mean, out_std,
            B, S, F,
            num_warps=1, num_stages=1
        )

        # Compute inverse-normal CDF for target_sparsity using Triton (scalar kernel)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        ndtri_scalar_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
            num_warps=1, num_stages=1
        )

        # Prepare output buffer (float32 for computation, cast later to bfloat16)
        out = torch.empty_like(inputs, dtype=torch.float32)

        # Apply threshold elementwise
        apply_threshold_kernel[grid](
            inputs, out_mean, out_std, out,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
