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

    # Iterate over D in tiles of 1024
    for off in range(0, D, 1024):
        idx = off + tl.arange(0, 1024)
        mask = idx < D
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    mean = acc_sum / D
    var_num = acc_sumsq / D - mean * mean
    var_num = tl.maximum(var_num, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var_num)

    tl.store(SUM_ptr + pid, acc_sum)
    tl.store(SUMSQ_ptr + pid, acc_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    acc_sum = tl.load(SUM_ptr + (b * S + s))
    acc_sumsq = tl.load(SUMSQ_ptr + (b * S + s))

    mean = acc_sum / D
    var_num = acc_sumsq / D - mean * mean
    var_num = tl.maximum(var_num, 0.0)
    std = tl.sqrt(var_num)

    tl.store(MEAN_ptr + (b * S + s), mean)
    tl.store(STD_ptr + (b * S + s), std)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
    P,               # float32 scalar target_sparsity
):
    # Abramowitz & Stegun 5.2.23 approximation (piecewise)
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

    # Select piecewise result
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_high = p > p_high

    # Since this is a scalar, we can evaluate branches and select
    res = tl.where(cond_low, result_low, 0.0)
    res = tl.where(cond_mid, result_mid, res)
    res = tl.where(cond_high, result_high, res)

    # Store scalar z-score
    tl.store(OUT_ptr, res)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous (float32)
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar icdf)
    OUT_ptr,         # *bfloat16, output [B, S, D]
    B, S, D,
    BLOCK_SIZE: tl.constexpr
):
    # 2D grid: axis 0 over (b, s), axis 1 over tiles of D
    pid0 = tl.program_id(axis=0)
    b = pid0 // S
    s = pid0 % S

    pid1 = tl.program_id(axis=1)
    off = pid1 * BLOCK_SIZE
    idx = off + tl.arange(0, BLOCK_SIZE)
    mask = idx < D

    # Load mean and std for this (b, s)
    mean = tl.load(MEAN_ptr + (b * S + s))
    std = tl.load(STD_ptr + (b * S + s))
    z = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z  # float32 scalar

    base = (b * S + s) * D

    # Load x, compute y = max(0, x - threshold)
    x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store as bfloat16
    tl.store(OUT_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only implementation: no torch ops on device tensors
        if target_sparsity == 0.0:
            # No sparsity requested, return input as bfloat16
            return input_tensor.to(torch.bfloat16)

        # Ensure contiguous input and compute in float32
        input_f32 = input_tensor.contiguous().to(torch.float32)
        B, S, D = input_f32.shape

        # Allocate buffers
        SUM = torch.empty(B * S, dtype=torch.float32, device=input_f32.device)
        SUMSQ = torch.empty(B * S, dtype=torch.float32, device=input_f32.device)
        MEAN = torch.empty(B * S, dtype=torch.float32, device=input_f32.device)
        STD = torch.empty(B * S, dtype=torch.float32, device=input_f32.device)
        Z = torch.empty(1, dtype=torch.float32, device=input_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](input_f32, SUM, SUMSQ, B, S, D, num_warps=4, num_stages=4)

        # Launch mean/std kernel: one program per (b, s)
        compute_mean_std_kernel[grid_reduce](SUM, SUMSQ, MEAN, STD, B, S, D, num_warps=4, num_stages=4)

        # Compute inverse-normal CDF for target_sparsity on device
        ndtri_approx_kernel[Z, target_sparsity](num_warps=1, num_stages=1)

        # Launch apply activation kernel: 2D grid over (b*s, tiles of D)
        BLOCK_SIZE = 1024
        grid_apply = (B * S, triton.cdiv(D, BLOCK_SIZE))
        OUT = torch.empty(B, S, D, dtype=torch.bfloat16, device=input_f32.device)
        apply_activation_kernel[grid_apply](input_f32, MEAN, STD, Z, OUT, B, S, D, BLOCK_SIZE, num_warps=8, num_stages=4)

        return OUT


def run(*args):
    return ModelNew()(*args)
