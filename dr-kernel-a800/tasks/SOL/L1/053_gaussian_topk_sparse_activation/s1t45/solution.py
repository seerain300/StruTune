import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    # Base offset for this row (contiguous layout: [B, S, D])
    base = (b * S + s) * D

    # Accumulators in float32
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Iterate over D in tiles
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    # Store results
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
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)

    # Compute mean and variance; clamp variance to non-negative
    mean = sum_val / D
    var_num = sumsq_val / D - mean * mean
    var_num = tl.maximum(var_num, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var_num)

    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
    P,                # float32 scalar (target_sparsity)
    A1, A2, A3, A4, A5, A6,
    B1, B2, B3, B4, B5,
    C1, C2, C3, C4, C5, C6,
    D1, D2, D3, D4,
    P_LOW, P_HIGH
):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF.
    # We compute into OUT_ptr[0].
    p_low = P_LOW
    p_high = P_HIGH

    # Masks for regions
    mask_low = P < p_low
    mask_mid = (P >= p_low) & (P <= p_high)
    mask_high = P > p_high

    # Compute z for each region
    z_low = 0.0
    z_mid = 0.0
    z_high = 0.0

    # Lower region: z = (a1 t + a2 t^2 + ...)/ (d1 t + d2 t^2 + ...)
    if mask_low:
        t = tl.sqrt(-2.0 * tl.log(P))
        num = (((((C1 * t + C2) * t + C3) * t + C4) * t + C5) * t + C6)
        den = (((((D1 * t + D2) * t + D3) * t + D4) * t + 1.0))
        z_low = num / den

    # Central region: z = (a1 r + a2 r^2 + ...)/ (b1 r + b2 r^2 + ...)
    if mask_mid:
        r = P - 0.5
        num = (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * r
        den = (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0)
        z_mid = num / den

    # Upper region: z = - (c1 q + c2 q^2 + ...)/ (d1 q + d2 q^2 + ...)
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - P))
        num = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
        den = (((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0))
        z_high = -num / den

    # Select z based on masks
    z = z_low
    if mask_mid:
        z = z_mid
    if mask_high:
        z = z_high

    # Store scalar result
    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bfloat16, output [B, S, D]
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr
):
    # 3D grid: axis 0 over (b, s), axis 1 over tiles of D
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
    z_score = tl.load(Z_ptr)  # scalar z

    # Compute threshold
    threshold = mean + std * z_score  # float32 scalar

    # Base offset for this row
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
        # Ensure input is contiguous
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # Shapes
        B, S, D = input_tensor.shape
        device = input_tensor.device

        # Allocate buffers (float32 for stats)
        sum_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        sumsq_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Prepare constants for inverse-normal approximation (A&S 5.2.23)
        a1, a2, a3, a4, a5, a6 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
        b1, b2, b3, b4, b5 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
        c1, c2, c3, c4, c5, c6 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
        d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        # Scalar z-score buffer (float32)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # 1) Reduce sum and sumsq per (b, s) row
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            input_tensor, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=1024, num_warps=4
        )

        # 2) Compute mean and std per (b, s) row
        compute_mean_std_kernel[grid_reduce](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D
        )

        # 3) Compute inverse-normal z-score for target sparsity (device-side scalar)
        p = float(target_sparsity)
        ndtri_approx_kernel[(1,)](
            z_buf, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high
        )
        z_score = z_buf[0]  # fetch scalar

        # 4) Apply activation: y = max(0, x - (mean + std * z_score)), store as bfloat16
        out = torch.empty((B, S, D), dtype=torch.bfloat16, device=device)
        grid_apply = (B * S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            input_tensor, mean_buf, std_buf, z_buf, out, B, S, D, BLOCK_SIZE=1024, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
