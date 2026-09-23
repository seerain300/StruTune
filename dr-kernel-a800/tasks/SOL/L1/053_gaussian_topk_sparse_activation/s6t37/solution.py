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
def ndtri_scalar_kernel(out_ptr, p: tl.float32, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low: tl.float32):
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
    # Lower region
    mask_low = (p > 0.0) & (p < p_low)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    res_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
              ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # Central region
    mask_mid = (p < 0.5) & (p >= p_low)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    res_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
              (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    # Upper region
    mask_high = (p <= 0.5) & (p > (1.0 - p_low))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    res_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
               ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    # Select result (only one branch will be true due to masks)
    res = tl.where(mask_low, res_low, 0.0)
    res = tl.where(mask_mid, res_mid, res)
    res = tl.where(mask_high, res_high, res)
    tl.store(out_ptr, res)


@triton.jit
def compute_thresholds_kernel(mean_ptr, std_ptr, out_thr_ptr, z: tl.float32, B, S):
    # One program per row (b, s)
    row = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    thr = mean + std * z
    tl.store(out_thr_ptr + row, thr)


@triton.jit
def apply_threshold_kernel(inp_ptr, thr_ptr, out_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(axis=0)
    b = row // S
    s = row % S
    base = b * stride_b + s * stride_s
    thr = tl.load(thr_ptr + row)  # per-row threshold (float32)
    start = 0
    while start < F:
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(inp_ptr + base + offs * stride_f, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs * stride_f, y, mask=mask)
        start += BLOCK_F


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Strides in elements
        stride_b, stride_s, stride_f = inputs.stride()

        # Choose block size for feature dimension
        if F >= 4096:
            BLOCK_F = 1024
            num_warps = 8
        elif F >= 1024:
            BLOCK_F = 512
            num_warps = 4
        else:
            BLOCK_F = 256
            num_warps = 4

        # 1) Compute per-row sum and sum of squares
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        sum_rows_kernel[(B * S,)](
            inputs, out_sum, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2
        )
        sumsq_rows_kernel[(B * S,)](
            inputs, out_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2
        )

        # 2) Compute mean and std per row (population std, unbiased=False)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[(B * S,)](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=1, num_stages=1)

        # 3) Compute inverse-normal CDF for scalar target_sparsity using A&S 7.1.26 in Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        ndtri_scalar_kernel[(1,)](
            z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low,
            num_warps=1, num_stages=1
        )
        z = z_buf[0]  # scalar z (float32)

        # 4) Compute per-row threshold = mean + std * z (float32)
        threshold = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_thresholds_kernel[(B * S,)](out_mean, out_std, threshold, z, B, S, num_warps=1, num_stages=1)

        # 5) Apply y = max(x - threshold[row], 0) elementwise
        out = torch.empty((B, S, F), dtype=torch.float32, device=device)
        apply_threshold_kernel[(B * S,)](
            inputs, threshold, out, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=2
        )

        # 6) Cast to bfloat16 to match original return dtype
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
