import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D          # int32 dimensions
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = (b * S + s) * D

    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    offset = 0
    while offset < D:
        idx = offset + tl.arange(0, 1024)
        mask = idx < D
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        offset += 1024

    tl.store(SUM_ptr + pid, acc_sum)
    tl.store(SUMSQ_ptr + pid, acc_sumsq)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
    P,               # float32 scalar target_sparsity
):
    # Abramowitz & Stegun 5.2.23 approximation
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
    p_high = 1.0 - p_low

    p = P  # float32 scalar

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                 ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    result_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                 (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                   ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Choose result based on p
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high
    res = tl.zeros((), dtype=tl.float32)
    res = tl.where(mask_low, result_low, res)
    res = tl.where(mask_mid, result_mid, res)
    res = tl.where(mask_high, result_high, res)

    tl.store(OUT_ptr, res)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    Z_ptr,           # *float32, length 1
    OUT_ptr,         # *bfloat16, output [B, S, D]
    B, S, D          # int32 dimensions
):
    # 2D grid: axis 0 over rows (B*S), axis 1 over tiles of D
    pid_row = tl.program_id(axis=0)
    b = pid_row // S
    s = pid_row % S

    base = (b * S + s) * D

    # Load sum and sumsq for this row
    sum_val = tl.load(SUM_ptr + pid_row)
    sumsq_val = tl.load(SUMSQ_ptr + pid_row)
    D_val = D  # int32 scalar
    mean = sum_val / D_val
    var_num = sumsq_val / D_val - mean * mean
    var_num = tl.maximum(var_num, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var_num)

    # Load z_score (scalar)
    z = tl.load(Z_ptr)

    threshold = mean + std * z  # float32 scalar

    tile = tl.program_id(axis=1)
    off = tile * 1024
    idx = off + tl.arange(0, 1024)
    mask = idx < D

    x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    tl.store(OUT_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only forward; no torch ops on device tensors.
        x = input_tensor.contiguous()
        B, S, D = x.shape
        device = x.device

        # Buffers for sums (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch reduction kernel: one program per (b, s) row
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            x, sum_buf, sumsq_buf, B, S, D,
            num_warps=8, num_stages=4
        )

        # Compute icdf(target_sparsity) on device (scalar)
        z_buf = torch.empty(1, dtype=torch.float32, device=device)
        # Pass target_sparsity as a 0-dim tensor to Triton kernel
        p_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
        ndtri_approx_kernel[(1,)](
            z_buf, p_tensor
        )

        # Output tensor (bfloat16)
        out = torch.empty_like(x, dtype=torch.bfloat16)

        # Launch activation kernel: 2D grid over rows and tiles of D
        grid_apply = (B * S, (D + 1023) // 1024)
        apply_activation_kernel[grid_apply](
            x, sum_buf, sumsq_buf, z_buf, out, B, S, D,
            num_warps=8, num_stages=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
