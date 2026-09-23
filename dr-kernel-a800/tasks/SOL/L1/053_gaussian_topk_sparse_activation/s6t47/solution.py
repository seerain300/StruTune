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
        ptr = x_ptr + row_b * stride_b + row_s * stride_s + offs * stride_f
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
    tl.store(out_sum_ptr + pid, total)


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
        ptr = x_ptr + row_b * stride_b + row_s * stride_s + offs * stride_f
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.store(out_sumsq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr,
                         B: tl.constexpr, S: tl.constexpr, F: tl.constexpr):
    pid = tl.program_id(axis=0)
    ssum = tl.load(out_sum_ptr + pid).to(tl.float32)
    ssq = tl.load(out_sumsq_ptr + pid).to(tl.float32)
    mean = ssum / F
    var = ssq / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(out_z_ptr, p: tl.float32,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        p_low: tl.float32):
    # Compute inverse normal CDF for p using Abramowitz & Stegun 7.1.26
    one_minus_p = 1.0 - p
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
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(one_minus_p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select region result
    z = tl.where(low_mask, z_low, 0.0) + tl.where(mid_mask, z_mid, 0.0) + tl.where(high_mask, z_high, 0.0)
    tl.store(out_z_ptr, z)


@triton.jit
def apply_threshold_kernel(x_ptr, mean_ptr, std_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
                           z: tl.float32, BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    row_b = pid // S
    row_s = pid % S
    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid).to(tl.float32)
    std = tl.load(std_ptr + pid).to(tl.float32)
    threshold = mean + std * z

    # Iterate over feature dimension, apply y = max(x - threshold, 0)
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        in_ptr = x_ptr + row_b * stride_b + row_s * stride_s + offs * stride_f
        x = tl.load(in_ptr, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        out_ptr_chunk = out_ptr + row_b * stride_b + row_s * stride_s + offs * stride_f
        tl.store(out_ptr_chunk, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA tensor
        assert inputs.is_cuda, "Inputs must be a CUDA tensor."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Strides for [B, S, F]
        stride_b = inputs.stride(0)
        stride_s = inputs.stride(1)
        stride_f = inputs.stride(2)

        # Heuristic for block size along F
        if F >= 8192:
            BLOCK_F = 1024
            num_warps = 4
        elif F >= 2048:
            BLOCK_F = 512
            num_warps = 4
        else:
            BLOCK_F = 256
            num_warps = 2

        # Allocate accumulators
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels: one program per row (b, s)
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, out_sum, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)
        sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Allocate mean and std buffers
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Compute mean and std per row
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=1, num_stages=1)

        # Compute inverse-normal CDF for scalar target_sparsity
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

        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity),
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4,
                                  p_low, num_warps=1, num_stages=1)
        z = z_buf[0]  # scalar

        # Prepare output buffer in float32
        out_f32 = torch.empty_like(inputs, dtype=torch.float32)

        # Apply threshold per row using correct strides
        apply_threshold_kernel[grid](inputs, out_mean, out_std, out_f32, B, S, F, stride_b, stride_s, stride_f, z, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Cast to bfloat16 to match original
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
