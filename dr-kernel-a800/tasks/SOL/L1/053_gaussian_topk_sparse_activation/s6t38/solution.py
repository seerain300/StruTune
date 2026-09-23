import math
import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(inp_ptr, out_sum_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(axis=0)
    b = row // S
    s = row % S
    base = b * stride_b + s * stride_s
    total = 0.0
    start = 0
    while start < F:
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(inp_ptr + base + offs * stride_f, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
        start += BLOCK_F
    tl.store(out_sum_ptr + row, total)


@triton.jit
def sumsq_rows_kernel(inp_ptr, out_sumsq_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(axis=0)
    b = row // S
    s = row % S
    base = b * stride_b + s * stride_s
    total = 0.0
    start = 0
    while start < F:
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(inp_ptr + base + offs * stride_f, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x * x, axis=0)
        start += BLOCK_F
    tl.store(out_sumsq_ptr + row, total)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr, B, S, F):
    row = tl.program_id(axis=0)
    sum_val = tl.load(out_sum_ptr + row)
    sumsq_val = tl.load(out_sumsq_ptr + row)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + row, mean)
    tl.store(out_std_ptr + row, std)


@triton.jit
def ndtri_scalar_kernel(out_ptr, p: tl.float32, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low):
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
    # z = inverse-phi(p)
    # Piecewise across low, mid, high regions using masks.
    # Note: This kernel writes to out_ptr[0].
    # Low region: p < p_low
    # Central region: p_low <= p <= (1 - p_low)
    # Upper region: p > (1 - p_low)
    # We compute the three branches and select per branch using masks.
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Constants
    a1v = a1; a2v = a2; a3v = a3; a4v = a4; a5v = a5; a6v = a6
    b1v = b1; b2v = b2; b3v = b3; b4v = b4; b5v = b5
    c1v = c1; c2v = c2; c3v = c3; c4v = c4; c5v = c5; c6v = c6
    d1v = d1; d2v = d2; d3v = d3; d4v = d4

    # Low region: q = sqrt(-2 log(p))
    # result_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    num_low = (((((c1v * q_low + c2v) * q_low + c3v) * q_low + c4v) * q_low + c5v) * q_low + c6v)
    den_low = ((((d1v * q_low + d2v) * q_low + d3v) * q_low + d4v) * q_low + 1.0)
    res_low = num_low / den_low

    # Mid region: q = p - 0.5
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1v * r_mid + a2v) * r_mid + a3v) * r_mid + a4v) * r_mid + a5v) * r_mid + a6v)
    den_mid = (((((b1v * r_mid + b2v) * r_mid + b3v) * r_mid + b4v) * r_mid + b5v) * r_mid + 1.0)
    res_mid = num_mid / den_mid

    # High region: q = sqrt(-2 log(1 - p))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    num_high = (((((c1v * q_high + c2v) * q_high + c3v) * q_high + c4v) * q_high + c5v) * q_high + c6v)
    den_high = ((((d1v * q_high + d2v) * q_high + d3v) * q_high + d4v) * q_high + 1.0)
    res_high = - (num_high / den_high)

    # Select result based on masks
    # Triton evaluates both branches, but selects with masks
    z = tl.where(mask_low, res_low, 0.0)
    z = tl.where(mask_mid, res_mid, z)
    z = tl.where(mask_high, res_high, z)
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(axis=0)
    b = row // S
    s = row % S
    base = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    threshold = mean + std  # multiplier is 1.0 as per original code, std is computed already with target_sparsity

    start = 0
    while start < F:
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(inp_ptr + base + offs * stride_f, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs * stride_f, y, mask=mask)
        start += BLOCK_F


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized forward:
        - Computes per-row mean and std along feature dim F (population std, unbiased=False).
        - Computes inverse-normal CDF at target_sparsity via A&S 7.1.26 in Triton.
        - Applies threshold: y = max(inputs - (mean + std * z), 0), returns bfloat16.
        """
        assert inputs.is_cuda, "ModelNew requires CUDA tensors for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Prepare accumulators
        out_sum = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Heuristic for block size
        if F >= 1024:
            BLOCK_F = 1024
            num_warps = 8
        elif F >= 256:
            BLOCK_F = 512
            num_warps = 4
        else:
            BLOCK_F = 256
            num_warps = 4

        # Launch reduction kernels: one program per row
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, out_sum, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)
        sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Compute per-row mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=1, num_stages=1)

        # Compute inverse-normal CDF for scalar p (use A&S 7.1.26)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
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

        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, num_warps=1, num_stages=1)
        z = z_buf[0]  # float32 scalar on device

        # Apply threshold: y = max(input - (mean + std * z), 0)
        out = torch.empty_like(inputs, dtype=torch.float32)
        apply_threshold_kernel[grid](inputs, out_mean, out_std, out, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2)

        # Return as bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
