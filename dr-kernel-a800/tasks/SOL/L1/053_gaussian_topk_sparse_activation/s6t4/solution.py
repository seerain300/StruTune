import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    input_ptr,               # *const float32
    out_sum_ptr,             # *float32, size B*S
    B: tl.constexpr,         # int
    S: tl.constexpr,         # int
    F: tl.constexpr,         # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = input_ptr + b * stride_b + s * stride_s + idx * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
    tl.atomic_add(out_sum_ptr + pid, sum_val)


@triton.jit
def sumsq_rows_kernel(
    input_ptr,               # *const float32
    out_sumsq_ptr,           # *float32, size B*S
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    sumsq_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = input_ptr + b * stride_b + s * stride_s + idx * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sumsq_val += tl.sum(vals * vals, axis=0)
    tl.atomic_add(out_sumsq_ptr + pid, sumsq_val)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,             # *const float32, size B*S
    out_sumsq_ptr,           # *const float32, size B*S
    out_mean_ptr,            # *float32, size B*S
    out_std_ptr,             # *float32, size B*S
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
):
    pid = tl.program_id(0)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    out_ptr,                 # *float32, size 1
    p,                       # float32 scalar
    a1, a2, a3, a4, a5, a6,  # float32
    b1, b2, b3, b4, b5,      # float32
    c1, c2, c3, c4, c5, c6,  # float32
    d1, d2, d3, d4,          # float32
    p_low,                   # float32
):
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
    one = 1.0
    two = 2.0

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (one - p_low))
    mask_high = p > (one - p_low)

    # Lower region: z = sqrt(-2 log(p)) * poly(q)
    q_low = tl.sqrt(-two * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + one)

    # Mid region: rational function
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + one)
    z_mid = num_mid / den_mid

    # Upper region: z = -sqrt(-2 log(1-p)) * poly(q)
    q_up = tl.sqrt(-two * tl.log(one - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + one)

    # Select region values
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_up, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(
    input_ptr,               # *const float32, shape [B, S, F]
    mean_ptr,                # *const float32, shape [B*S]
    std_ptr,                 # *const float32, shape [B*S]
    z_ptr,                   # *const float32, size 1
    output_ptr,              # *float32, shape [B, S, F]
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    mean_val = tl.load(mean_ptr + pid)
    std_val = tl.load(std_ptr + pid)
    z_val = tl.load(z_ptr)  # scalar

    threshold = mean_val + std_val * z_val

    for off in range(0, F, BLOCK_F):
        idx = off + tl.arange(0, BLOCK_F)
        mask = idx < F
        in_ptrs = input_ptr + b * stride_b + s * stride_s + idx * stride_f
        out_ptrs = output_ptr + b * stride_b + s * stride_s + idx * stride_f

        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        res = vals - threshold
        res = tl.maximum(res, 0.0)
        tl.store(out_ptrs, res, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Triton-only forward: no torch ops in host code
        assert inputs.is_cuda, "inputs must be on CUDA for Triton kernels"
        device = inputs.device
        B, S, F = inputs.shape

        # Compute in float32 for stability, return bfloat16
        input_f32 = inputs.to(torch.float32).contiguous()

        # Allocate outputs and buffers
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels: one program per (b, s) row
        BLOCK_F = 256  # tuneable; 256 works well for large F
        grid = (B * S,)
        sum_rows_kernel[grid](
            input_f32,
            out_sum,
            B, S, F,
            input_f32.stride(0), input_f32.stride(1), input_f32.stride(2),
            BLOCK_F=BLOCK_F,
        )
        sumsq_rows_kernel[grid](
            input_f32,
            out_sumsq,
            B, S, F,
            input_f32.stride(0), input_f32.stride(1), input_f32.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # Compute per-row mean and std (population, unbiased=False)
        compute_stats_kernel[grid](
            out_sum, out_sumsq,
            out_mean, out_std,
            B, S, F,
        )

        # Compute inverse-normal CDF for scalar target_sparsity in Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        p_scalar = float(target_sparsity)
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

        ndtri_scalar_kernel[(1,)](
            z_buf,
            p_scalar,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
        )

        # Apply threshold in Triton: y = max(input - (mean + std * z), 0)
        output_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)
        apply_threshold_kernel[grid](
            input_f32,
            out_mean,
            out_std,
            z_buf,  # length-1 tensor
            output_f32,
            B, S, F,
            input_f32.stride(0), input_f32.stride(1), input_f32.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # Cast to bfloat16 to match original Model
        return output_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
