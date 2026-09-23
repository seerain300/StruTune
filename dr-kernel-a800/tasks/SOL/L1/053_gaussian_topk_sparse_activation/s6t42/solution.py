import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(inputs_ptr, out_sum_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc = 0.0  # float32 accumulator
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
        offs += BLOCK_F
    tl.atomic_add(out_sum_ptr + pid, acc)


@triton.jit
def sumsq_rows_kernel(inputs_ptr, out_sumsq_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc = 0.0  # float32 accumulator
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        x2 = x * x
        acc += tl.sum(x2, axis=0)
        offs += BLOCK_F
    tl.atomic_add(out_sumsq_ptr + pid, acc)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr, B, S, F):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(0)
    sum_val = tl.load(out_sum_ptr + pid).to(tl.float32)
    sumsq_val = tl.load(out_sumsq_ptr + pid).to(tl.float32)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    # Ensure non-negative variance to avoid tiny negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(out_ptr, p: tl.float32):
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
    # Piecewise regions for better accuracy
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Region 1: p < p_low
    # Use erf-based approximation: z ~ sqrt(2) * (1 - erf((1 - p)/2) / sqrt(pi))
    # Triton has tl.erf; we implement via built-in if available. Here we use a simple polynomial if not.
    # We'll implement the standard polynomial approximation:
    # z = sign(p - 0.5) * (1.0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5), t = sqrt(2x) with x = p
    # However, Triton does not provide tl.erf in some builds; use piecewise polynomial approximation.
    # We'll use the well-known Hastings approximation for erf:
    # erf(x) ≈ sign(x) * (1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-x^2)), t = 1 / (1 + p*0.5)
    # But to keep simplicity and portability, we implement the A&S polynomial approximation directly:
    # For z(x), with x = p, use z = sign(x - 0.5) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5), t = sqrt(2x)
    # Coefficients:
    a1 = 0.254829592; a2 = -0.284496736; a3 = 1.421413741; a4 = -1.453152027; a5 = 1.061405429
    x = p
    px = abs(x - 0.5)
    t = tl.sqrt(2.0 * px)
    # Horner evaluation for polynomial
    poly = a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))
    # sign for erf-like: since erf(x) = 1 - 2/(sqrt(pi))*exp(-x^2) for small x, and sign here, we use sign(p - 0.5)
    sign = 1.0 if (x > 0.5) else -1.0
    z1 = sign * poly

    # Region 2: p >= p_low and p <= p_high: use standard A&S 7.1.26 polynomial
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01

    x = p - 0.5
    r = x * x
    poly_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * x
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z2 = poly_mid / denom

    # Region 3: p > p_high: symmetry z = -z1 (using the same polynomial for 1 - p)
    p_high_mask = p > p_high
    z_high = -z1  # symmetry for the tail region

    # Select final z: regions via masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)

    # Default to mid, override low/high
    z = z2
    z = tl.where(mask_low, z1, z)
    z = tl.where(mask_high, z_high, z)  # mask_high is p > p_high

    # Store
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(inputs_ptr, out_ptr, mean_ptr, std_ptr, z_val: tl.float32, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid).to(tl.float32)
    std = tl.load(std_ptr + pid).to(tl.float32)
    threshold = mean + std * z_val

    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs_in = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs_in, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        ptrs_out = out_ptr + base + idx * stride_f
        tl.store(ptrs_out, y, mask=mask)
        offs += BLOCK_F


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run:
        - Computes per-(batch, seq) mean and std over feature dim (population std, unbiased=False).
        - Uses inverse-normal CDF (Abramowitz & Stegun) for z = ndtri(target_sparsity).
        - Applies y = max(input - (mean + std*z), 0) and returns bfloat16.
        All computation happens in Triton kernels; no torch ops in forward.
        """
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Output buffer for elementwise result (float32 for kernel writes)
        out = torch.empty_like(inputs, dtype=torch.float32, device=device)

        # Strides
        stride_b, stride_s, stride_f = inputs.stride()

        # Choose BLOCK_F based on F
        if F >= 1024:
            BLOCK_F = 1024
        elif F >= 512:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # Allocate accumulators for sum and sumsq (one per row)
        row_sums = torch.zeros((B * S,), dtype=torch.float32, device=device)
        row_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels: one program per row
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, row_sums, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)
        sumsq_rows_kernel[grid](inputs, row_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Compute mean and std per row
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](row_sums, row_sumsq, out_mean, out_std, B, S, F)

        # Compute inverse-normal CDF for target_sparsity as a scalar on device
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity))

        # Apply threshold: y = max(input - (mean + std*z), 0)
        apply_threshold_kernel[grid](inputs, out, out_mean, out_std, z_buf[0], B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Cast to bfloat16 to match original return type
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
