import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(inp_ptr, out_sum_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    # Pointer to the start of the row (b, s, :)
    row_ptr = inp_ptr + b * stride_b + s * stride_s
    total = tl.zeros((), dtype=tl.float32)
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(row_ptr + idx * stride_f, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        total += tl.sum(vals, axis=0)
    tl.atomic_add(out_sum_ptr + pid, total)


@triton.jit
def sumsq_rows_kernel(inp_ptr, out_sumsq_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    row_ptr = inp_ptr + b * stride_b + s * stride_s
    total = tl.zeros((), dtype=tl.float32)
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(row_ptr + idx * stride_f, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        total += tl.sum(vals * vals, axis=0)
    tl.atomic_add(out_sumsq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(sum_ptr, sumsq_ptr, out_mean_ptr, out_std_ptr, B, S, F):
    pid = tl.program_id(0)
    total = tl.load(sum_ptr + pid)
    totalsq = tl.load(sumsq_ptr + pid)
    mean = total / F
    var = totalsq / F - mean * mean
    # clamp variance to non-negative for numerical stability
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(z_ptr, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low):
    # Compute z = inverse-normal CDF for p using Abramowitz & Stegun 7.1.26 approximation
    # Region masks
    low_mask = p > 0.0 and p < p_low
    mid_mask = p >= p_low and p <= (1.0 - p_low)
    high_mask = not (low_mask or mid_mask)  # p >= 1 - p_low, note p is in (0, 1)
    # epsilon for safe log
    eps = 1e-7
    q_low = tl.sqrt(-2.0 * tl.log(p + (1.0 - p_low) * eps))
    z_low = -(((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5 * q_low + c6
    denom_low = (((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0
    z_low = z_low / denom_low

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = z_mid / denom_mid

    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p + p_low * eps))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = z_high / denom_high

    # Select by masks
    z = tl.where(low_mask, z_low, 0.0)
    z = tl.where(mid_mask, z_mid, z)
    z = tl.where(high_mask, z_high, z)
    tl.store(z_ptr, z)


@triton.jit
def apply_threshold_kernel(inp_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    row_ptr = inp_ptr + b * stride_b + s * stride_s
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    threshold = mean + std * z
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(row_ptr + idx * stride_f, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + b * stride_b + s * stride_s + idx * stride_f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only implementation: no torch ops in forward
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Allocate per-row accumulators
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Choose block size
        BLOCK_F = 256  # good starting point; tune as needed

        # Launch reduction kernels
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, out_sum, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=4)
        sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=4)

        # Allocate outputs for mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Compute mean and std per row
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=4)

        # Allocate buffer for inverse-normal CDF result (scalar)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # Inverse-normal CDF for target_sparsity using A&S 7.1.26
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, num_warps=1)

        # Elementwise thresholding and ReLU
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)
        apply_threshold_kernel[grid](inputs, out_mean, out_std, z_buf, out_f32, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=4)

        # Cast to bfloat16 to match original Model’s output dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
