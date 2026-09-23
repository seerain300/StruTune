import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_rows_kernel(
    inputs_ptr,          # *const T, will load as fp32
    out_mean_ptr,        # *fp32, size B*S
    out_std_ptr,         # *fp32, size B*S
    B: tl.constexpr,     # batch size
    S: tl.constexpr,     # seq_len
    F: tl.constexpr,     # intermediate_size (feature dim)
    x_stride_b, x_stride_s, x_stride_f,
    BLOCK_F: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S

    row_base = inputs_ptr + b * x_stride_b + s * x_stride_s

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over feature dimension in chunks of BLOCK_F
    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(row_base + offs * x_stride_f, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute population mean and std
    F_f32 = tl.full((), F, tl.float32)
    mean = sum_val / F_f32
    var = (sumsq_val / F_f32) - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # Store per-row stats
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def compute_threshold_kernel(
    out_mean_ptr,     # *fp32
    out_std_ptr,      # *fp32
    out_threshold_ptr,# *fp32, size B*S
    z,                # fp32 scalar (inverse-normal CDF of target_sparsity)
    B: tl.constexpr,
    S: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    mean = tl.load(out_mean_ptr + pid)
    std = tl.load(out_std_ptr + pid)
    thr = mean + std * z
    tl.store(out_threshold_ptr + pid, thr)


@triton.jit
def apply_threshold_kernel(
    inputs_ptr,             # *const fp32 (we load as fp32)
    threshold_ptr,          # *fp32, size B*S
    out_ptr,                # *fp32 (store fp32, host casts to bfloat16)
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    x_stride_b, x_stride_s, x_stride_f,
    y_stride_b, y_stride_s, y_stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S

    row_inputs = inputs_ptr + b * x_stride_b + s * x_stride_s
    row_out = out_ptr + b * y_stride_b + s * y_stride_s
    thr = tl.load(threshold_ptr + pid)  # scalar per (b, s)

    for f_start in range(0, F, BLOCK_F):
        offs = f_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(row_inputs + offs * x_stride_f, mask=mask, other=0.0)  # fp32
        y = tl.maximum(x - thr, 0.0)
        tl.store(row_out + offs * y_stride_f, y, mask=mask)


@triton.jit
def ndtri_scalar_kernel(
    out_ptr,      # *fp32, size 1
    p,            # fp32 scalar
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low: tl.constexpr,
):
    # A&S 7.1.26 approximation
    q_low = tl.sqrt(2.0 * tl.log(p))
    num_low = ((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6
    den_low = ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = num_low / den_low

    q_high = tl.sqrt(2.0 * tl.log(1.0 - p))
    num_high = ((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6
    den_high = ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = -num_high / den_high

    q_mid = p - 0.5
    q2 = q_mid * q_mid
    num_mid = (((((a1 * q2 + a2) * q2 + a3) * q2 + a4) * q2 + a5) * q2 + a6) * q_mid
    den_mid = (((((b1 * q2 + b2) * q2 + b3) * q2 + b4) * q2 + b5) * q2 + 1.0)
    z_mid = num_mid / den_mid

    mask_low = p < p_low
    mask_high = p > (1.0 - p_low)
    z = z_mid
    z = tl.where(mask_low, z_low, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(out_ptr, z)


def _choose_block_f(F: int) -> int:
    # Heuristic: power-of-two block up to 1024 for good occupancy
    if F >= 2048:
        return 1024
    elif F >= 1024:
        return 1024
    elif F >= 512:
        return 512
    else:
        return 256


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs

        assert inputs.is_cuda, "Inputs must be on CUDA for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Output in fp32 for elementwise math; cast to bfloat16 at end
        out = torch.empty_like(inputs, dtype=torch.float32)

        # Per-row stats (mean, std) in fp32
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        x_stride_b, x_stride_s, x_stride_f = inputs.stride()
        y_stride_b, y_stride_s, y_stride_f = out.stride()

        BLOCK_F = _choose_block_f(F)
        grid = (B * S,)

        # Single pass reduction: sum and sumsq per row
        sum_sumsq_rows_kernel[grid](
            inputs, out_mean, out_std,
            B, S, F,
            x_stride_b, x_stride_s, x_stride_f,
            BLOCK_F=BLOCK_F,
        )

        # Compute inverse-normal CDF for target_sparsity
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        ndtri_scalar_kernel[(1,)](
            z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low=p_low,
        )
        z = z_buf[0]  # scalar fp32

        # Compute threshold per (b, s): mean + std * z
        threshold = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_threshold_kernel[grid](
            out_mean, out_std, threshold, z, B, S
        )

        # Apply threshold: y = max(input - threshold[b,s], 0) elementwise along F
        apply_threshold_kernel[grid](
            inputs, threshold, out,
            B, S, F,
            x_stride_b, x_stride_s, x_stride_f,
            y_stride_b, y_stride_s, y_stride_f,
            BLOCK_F=BLOCK_F,
        )

        # Cast back to bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
