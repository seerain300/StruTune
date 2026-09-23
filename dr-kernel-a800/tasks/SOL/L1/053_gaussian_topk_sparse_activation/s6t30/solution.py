import torch
import triton
import triton.language as tl


@triton.jit
def reduce_rows_fused_kernel(
    x_ptr,                # *const float32
    out_sum_ptr,          # *float32
    out_sumsq_ptr,        # *float32
    B: tl.constexpr,      # int
    S: tl.constexpr,      # int
    F: tl.constexpr,      # int
    stride_b,             # int (elements)
    stride_s,             # int (elements)
    stride_f,             # int (elements)
    BLOCK_F: tl.constexpr # int
):
    pid = tl.program_id(0)  # 0 .. (B*S - 1)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over the feature dimension in chunks of BLOCK_F
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = x_ptr + base + idx * stride_f
        x_vec = tl.load(ptrs, mask=mask, other=0.0)
        x_vec = x_vec.to(tl.float32)
        sum_val += tl.sum(x_vec, axis=0)
        sumsq_val += tl.sum(x_vec * x_vec, axis=0)

    # Store per-row sums
    tl.atomic_add(out_sum_ptr + pid, sum_val)
    tl.atomic_add(out_sumsq_ptr + pid, sumsq_val)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,       # *const float32
    out_sumsq_ptr,     # *const float32
    out_mean_ptr,      # *float32
    out_std_ptr,       # *float32
    B: tl.constexpr,   # int
    S: tl.constexpr,   # int
    F: tl.constexpr    # int
):
    pid = tl.program_id(0)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    F_f = tl.full((), F, tl.float32)
    mean = sum_val / F_f
    var = sumsq_val / F_f - mean * mean
    # clamp to non-negative to avoid small negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    out_ptr,           # *float32, length 1
    p,                 # float32 scalar
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low              # float32
):
    # Abramowitz & Stegun 7.1.26
    mask_low = p < p_low
    mask_high = p > (1.0 - p_low)
    mask_mid = ~mask_low & ~mask_high

    # Low region: p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / den_low

    # Mid region: p_low <= p <= 1 - p_low
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # High region: p > 1 - p_low
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / den_high

    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_high, 0.0)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(
    inputs_ptr,         # *const float32
    mean_ptr,           # *const float32, length B*S
    std_ptr,            # *const float32, length B*S
    z_scalar,           # float32 scalar
    out_ptr,            # *float32
    B: tl.constexpr,    # int
    S: tl.constexpr,    # int
    F: tl.constexpr,    # int
    stride_b,           # int
    stride_s,           # int
    stride_f,           # int
    BLOCK_F: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * z_scalar

    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        in_ptrs = inputs_ptr + base + idx * stride_f
        out_ptrs = out_ptr + base + idx * stride_f

        x = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only forward. Inputs: [B, S, F].
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        # Ensure contiguous layout
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Choose block size for feature dimension
        if F >= 16384:
            BLOCK_F = 1024
        elif F >= 8192:
            BLOCK_F = 1024
        elif F >= 4096:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # 1) Fused reduction: per-row sum and sumsq
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        grid = (B * S,)
        reduce_rows_fused_kernel[grid](
            inputs, out_sum, out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=4
        )

        # 2) Compute per-row mean and std (population std)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        compute_stats_kernel[grid](
            out_sum, out_sumsq, out_mean, out_std,
            B, S, F,
            num_warps=4
        )

        # 3) Compute inverse-normal CDF for target_sparsity (A&S 7.1.26) in Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        # Constants (Abramowitz & Stegun 7.1.26)
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
            num_warps=1
        )
        z_scalar = float(z_buf.item())

        # 4) Apply threshold: y = max(input - (mean + std * z), 0)
        inputs_f32 = inputs.to(torch.float32)
        out_f32 = torch.empty_like(inputs_f32)

        apply_threshold_kernel[grid](
            inputs_f32, out_mean, out_std, z_scalar,
            out_f32,
            B, S, F,
            inputs_f32.stride(0), inputs_f32.stride(1), inputs_f32.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=4
        )

        # 5) Cast back to bfloat16 to match original model's return dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
