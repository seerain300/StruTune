import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    x_ptr,          # *float32, input tensor as float32
    out_sum_ptr,    # *float32, per-row sum
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # One program per row (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_sum = 0.0
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptr = x_ptr + b * stride_b + s * stride_s + offs * stride_f
        vals = tl.load(ptr, mask=mask, other=0.0)
        row_sum += tl.sum(vals, axis=0)
    tl.store(out_sum_ptr + pid, row_sum)


@triton.jit
def sumsq_rows_kernel(
    x_ptr,          # *float32, input tensor as float32
    out_sumsq_ptr,  # *float32, per-row sum of squares
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # One program per row (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_sumsq = 0.0
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptr = x_ptr + b * stride_b + s * stride_s + offs * stride_f
        vals = tl.load(ptr, mask=mask, other=0.0)
        row_sumsq += tl.sum(vals * vals, axis=0)
    tl.store(out_sumsq_ptr + pid, row_sumsq)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,    # *float32
    out_sumsq_ptr,  # *float32
    out_mean_ptr,   # *float32
    out_std_ptr,    # *float32
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr
):
    # One program per row
    pid = tl.program_id(0)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    out_z_ptr,          # *float32, 1-element buffer to store z
    p,                  # float32 scalar target sparsity
    a1, a2, a3, a4, a5, a6,  # A&S constants
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low
):
    # Inverse-normal CDF via Abramowitz & Stegun 7.1.26
    t = 0.7071067811865476  # 1/sqrt(2)

    # Lower region: p < p_low
    x_lower = -1.4142135623730951 * tl.sqrt(tl.log(p))  # -sqrt(2) * sqrt(log(p))
    q_lower = tl.sqrt(-2.0 * tl.log(p))

    # Mid region: p in [p_low, 1 - p_low]
    q_mid = p - 0.5

    # Upper region: p > 1 - p_low
    x_upper = 1.4142135623730951 * tl.sqrt(tl.log(1.0 - p))  # +sqrt(2) * sqrt(log(1-p))
    q_upper = tl.sqrt(-2.0 * tl.log(1.0 - p))

    # Lower polynomial
    q2 = q_lower * q_lower
    c_poly_lower = (((c1 * q_lower + c2) * q_lower + c3) * q_lower + c4) * q_lower + c5
    c_poly_lower = c_poly_lower * q_lower + c6
    z_lower = x_lower + (q2 + 1.0) * q_lower * c_poly_lower

    # Mid polynomial
    b_poly_mid = (((b1 * q_mid + b2) * q_mid + b3) * q_mid + b4) * q_mid + b5
    a_poly_mid = (((a1 * q_mid + a2) * q_mid + a3) * q_mid + a4) * q_mid + a5
    a_poly_mid = a_poly_mid * q_mid + a6
    z_mid = (t * t + 1.0) * q_mid * a_poly_mid * q_mid  # q_mid squared cancels

    # Upper polynomial
    q2 = q_upper * q_upper
    c_poly_upper = (((c1 * q_upper + c2) * q_upper + c3) * q_upper + c4) * q_upper + c5
    c_poly_upper = c_poly_upper * q_upper + c6
    z_upper = x_upper + (q2 + 1.0) * q_upper * c_poly_upper

    # Select by region
    # Note: Triton supports tl.where with scalar conditions; we compare p to scalars
    z = tl.where(p < p_low, z_lower, z_mid)
    z = tl.where(p > (1.0 - p_low), z_upper, z)

    tl.store(out_z_ptr, z)


@triton.jit
def apply_threshold_kernel(
    x_ptr,           # *float32, input as float32
    out_ptr,         # *float32, output as float32
    mean_ptr,        # *float32, per-row mean
    std_ptr,         # *float32, per-row std
    z_ptr,           # *float32, scalar z (1-element)
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # One program per row (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar

    threshold = mean + std * z
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        in_ptr = x_ptr + b * stride_b + s * stride_s + offs * stride_f
        out_ptr_row = out_ptr + b * stride_b + s * stride_s + offs * stride_f
        vals = tl.load(in_ptr, mask=mask, other=0.0)
        y = vals - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr_row, y)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized sparse activation:
        - Compute per-(batch, seq) mean and std along the feature dimension (F).
        - z = inverse_normal_cdf(target_sparsity) via A&S 7.1.26 approximation in Triton.
        - threshold = mean + std * z
        - y = max(input - threshold, 0), return bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous
        if not inputs.is_cuda:
            inputs = inputs.to(device="cuda")
        inputs = inputs.contiguous()

        # Work in float32 for accumulation
        x = inputs.to(torch.float32)

        B, S, F = x.shape
        device = x.device

        # Strides for (B, S, F)
        stride_b = S * F
        stride_s = F
        stride_f = 1

        # Output buffer for elementwise result (float32)
        y = torch.empty_like(x, dtype=torch.float32)

        # Choose block size for feature loop
        BLOCK_F = 1024 if F >= 1024 else (512 if F >= 512 else 256)
        NUM_WARPS = 4 if BLOCK_F <= 1024 else 8

        # Compute per-row sum and sumsq (one program per row)
        out_sum = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=device)

        grid = (B * S,)
        sum_rows_kernel[grid](x, out_sum, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=NUM_WARPS, num_stages=2)
        sumsq_rows_kernel[grid](x, out_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F, num_warps=NUM_WARPS, num_stages=2)

        # Compute mean and std per row
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=1, num_stages=1)

        # Compute inverse-normal CDF for target_sparsity
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
        z = z_buf[0]  # scalar float32

        # Apply threshold per row: y = max(x - (mean + std * z), 0)
        apply_threshold_kernel[grid](
            x, y, out_mean, out_std, z,
            B, S, F,
            stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
            num_warps=NUM_WARPS, num_stages=2
        )

        # Cast back to bfloat16 to match original
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
