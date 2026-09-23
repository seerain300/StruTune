import torch
import triton
import triton.language as tl


# -----------------------
# Triton kernels
# -----------------------

@triton.jit
def sum_rows_kernel(X_ptr, out_sum_ptr,
                    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                    stride_b, stride_s, stride_f,
                    BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    row_start = b * stride_b + s * stride_s

    total_sum = 0.0
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
    tl.store(out_sum_ptr + pid, total_sum)


@triton.jit
def sumsq_rows_kernel(X_ptr, out_sumsq_ptr,
                      B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                      stride_b, stride_s, stride_f,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    row_start = b * stride_b + s * stride_s

    total_sumsq = 0.0
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        total_sumsq += tl.sum(x * x, axis=0)
    tl.store(out_sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr,
                         B: tl.constexpr, S: tl.constexpr, F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)

    F_f = tl.float32(F)
    mean = sum_val / F_f
    var = sumsq_val / F_f - mean * mean
    var = tl.maximum(var, 0.0)  # clamp to non-negative
    std = tl.sqrt(var)

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(out_ptr, p,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4, p_low):
    # Compute inverse-normal CDF for p using A&S 7.1.26 approximation.
    # out_ptr[0] will hold the result.
    p = tl.float32(p)
    one = 1.0
    pi = 3.141592653589793
    p_low = tl.float32(p_low)

    # Regions
    low_mask = p < p_low
    mid_mask = (p >= p_low) & (p <= (1.0 - p_low))
    high_mask = p > (1.0 - p_low)

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Mid region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_a = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_b = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_a * q_mid / poly_b

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select region result
    z = tl.zeros((), dtype=tl.float32)
    z = tl.where(low_mask, z_low, z)
    z = tl.where(mid_mask, z_mid, z)
    z = tl.where(high_mask, z_high, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, mean_ptr, std_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b, stride_s, stride_f):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    row_start = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    # z is a scalar (precomputed). We'll load it from a small buffer via a pointer.
    z = tl.load(out_ptr)  # scalar z from ndtri kernel

    threshold = mean + std * z
    for off in range(0, F, 1024):  # chunk over feature dimension
        cols = off + tl.arange(0, 1024)
        mask = cols < F
        x_ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        # Linearize output per row: out[pid, cols] stores contiguous
        out_row_ptr = out_ptr + pid * F
        out_ptrs = out_row_ptr + cols
        tl.store(out_ptrs, y, mask=mask)


# -----------------------
# Host-side functions
# -----------------------

@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size] (F on last dim).
        target_sparsity: Float in [0, 1] indicating target sparsity level.

    Returns:
        Sparsified tensor of same shape as input, dtype bfloat16.
    """
    assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
    inputs = inputs.contiguous()
    B, S, F = inputs.shape
    device = inputs.device

    # Prepare per-row accumulators
    out_sum = torch.empty((B * S,), dtype=torch.float32, device=device)
    out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=device)
    out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
    out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

    # Choose block size for feature dimension
    if F >= 4096:
        BLOCK_F = 1024
    elif F >= 1024:
        BLOCK_F = 512
    else:
        BLOCK_F = 256

    # Compute sums and sumsq across F (one program per row)
    grid = (B * S,)
    sum_rows_kernel[grid](inputs, out_sum, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F)
    sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F)

    # Compute per-row mean and std (population)
    compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F)

    # Compute z = inverse-normal CDF for target_sparsity (A&S 7.1.26) using Triton
    z_buf = torch.empty((1,), dtype=torch.float32, device=device)
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
    p_low = 0.02425
    ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low)

    z = z_buf[0]  # scalar z on device

    # Apply threshold: output = max(input - (mean + std * z), 0)
    out_f32 = torch.empty_like(inputs, dtype=torch.float32)
    apply_threshold_kernel[(B * S,)](inputs, out_mean, out_std, z, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2))

    # Cast to bfloat16 to match original return dtype
    return out_f32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor input.")
        return run(*args)


def run(*args):
    return ModelNew()(*args)
