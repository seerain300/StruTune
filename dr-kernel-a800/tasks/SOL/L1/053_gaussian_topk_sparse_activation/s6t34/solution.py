import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    x_ptr,               # *const float32
    out_sum_ptr,         # *float32, length B*S
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    BLOCK_F: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (b, s) row
    sum_val = 0.0
    # Iterate over feature dimension in chunks
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        # For contiguous input: stride_f = 1
        x_vec = tl.load(x_ptr + pid * F + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vec, axis=0)
    tl.store(out_sum_ptr + pid, sum_val)


@triton.jit
def sumsq_rows_kernel(
    x_ptr,               # *const float32
    out_sumsq_ptr,       # *float32, length B*S
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    BLOCK_F: tl.constexpr
):
    pid = tl.program_id(0)
    sumsq_val = 0.0
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x_vec = tl.load(x_ptr + pid * F + offs, mask=mask, other=0.0).to(tl.float32)
        sumsq_val += tl.sum(x_vec * x_vec, axis=0)
    tl.store(out_sumsq_ptr + pid, sumsq_val)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,         # *const float32
    out_sumsq_ptr,       # *const float32
    out_mean_ptr,        # *float32
    out_std_ptr,         # *float32
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr
):
    pid = tl.program_id(0)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    F_f = tl.full((), F, tl.float32)
    mean = sum_val / F_f
    var = sumsq_val / F_f - mean * mean
    # Clamp to non-negative to avoid small negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    out_ptr,             # *float32, length 1
    p,                   # float32 scalar
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low                # float32 scalar
):
    # Abramowitz & Stegun 7.1.26 approximation
    mask_low = p < p_low
    mask_high = p > (1.0 - p_low)
    mask_mid = ~mask_low & ~mask_high

    # Low region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / den_low

    # Mid region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # High region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / den_high  # negative sign for x > 0.5

    # Combine regions
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(
    in_ptr,              # *const float32
    mean_ptr,            # *const float32, length B*S
    std_ptr,             # *const float32, length B*S
    z_ptr,               # *const float32, length 1
    out_ptr,             # *float32
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    BLOCK_F: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (b, s) row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)
    threshold = mean + std * z
    base_in = in_ptr + pid * F
    base_out = out_ptr + pid * F
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(base_in + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        # y = max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(base_out + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure device and dtype handling: Triton requires CUDA tensors
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        # Make input contiguous along the last dim for simple addressing
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape

        # Choose block size for feature iteration
        # Use a power-of-two up to 1024, tuned for typical F sizes
        if F >= 2048:
            BLOCK_F = 1024
        elif F >= 1024:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # Allocate accumulators for sum and sumsq
        out_sum = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Launch reduction kernels: one program per (b, s) row
        grid = (B * S,)
        sum_rows_kernel[grid](x, out_sum, B, S, F, BLOCK_F)
        sumsq_rows_kernel[grid](x, out_sumsq, B, S, F, BLOCK_F)

        # Compute per-row mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F)

        # Compute inverse-normal CDF for scalar target_sparsity using Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=x.device)
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
        ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low)

        # Apply threshold: y = max(x - (mean + std * z), 0)
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)
        apply_threshold_kernel[grid](x, out_mean, out_std, z_buf, y, B, S, F, BLOCK_F)

        # Return bfloat16 to match original model
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
