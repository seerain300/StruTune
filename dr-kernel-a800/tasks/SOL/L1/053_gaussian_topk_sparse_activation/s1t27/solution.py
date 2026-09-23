import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor, arbitrary 3D layout
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    stride_x0, stride_x1, stride_x2,  # strides for [B, S, D] in elements
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Compute base offset for this (b, s) row
    base = b * stride_x0 + s * stride_x1

    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    num_chunks = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        offs = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        idx = base + offs * stride_x2
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, total_sum)
    tl.store(SUMSQ_ptr + pid, total_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32 number of features
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF
    p = tl.load(P_ptr)
    # Constants
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

    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / den_low

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5))
    z_mid = poly_mid / (den_mid + 1.0)

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / den_high

    # Select region
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor, arbitrary 3D layout
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bfloat16 output [B, S, D]
    B, S, D,         # int32
    stride_x0, stride_x1, stride_x2,  # strides for [B, S, D] in elements
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * stride_x0 + s * stride_x1
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    idx = base + offs * stride_x2

    mean = tl.load(MEAN_ptr + b * S + s)
    std = tl.load(STD_ptr + b * S + s)
    z_score = tl.load(Z_ptr)
    threshold = mean + std * z_score

    x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure inputs are on device and 3D
        if not inputs.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor inputs.")
        if inputs.dim() != 3:
            raise RuntimeError("ModelNew expects a 3D tensor [B, S, D].")
        B, S, D = inputs.shape
        x = inputs.contiguous()

        # Allocate buffers for sums in float32
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel over features using element strides
        BLOCK_SIZE = 1024
        grid = (B * S,)
        reduce_sum_sumsq_kernel[grid](
            x, sum_buf, sumsq_buf,
            B, S, D,
            x.stride(0), x.stride(1), x.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        # Allocate mean/std buffers
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch compute mean/std kernel
        compute_mean_std_kernel[grid](
            sum_buf, sumsq_buf, mean_buf, std_buf,
            D,
            num_warps=4,
        )

        # Compute z_score via Triton ndtri approximation (one scalar)
        p_tensor = torch.empty(1, dtype=torch.float32, device=x.device)
        p_tensor[0] = float(target_sparsity)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_approx_kernel[(1,)](
            p_tensor, z_buf,
            num_warps=1,
        )

        # Allocate output in bfloat16 as per original function’s return dtype
        out = torch.empty((B, S, D), dtype=torch.bfloat16, device=x.device)

        # Launch apply activation kernel (3D grid over (B, S, tiles of D))
        apply_activation_kernel[(B, S, triton.cdiv(D, BLOCK_SIZE))](  # 3D grid
            x, mean_buf, std_buf, z_buf, out,
            B, S, D,
            x.stride(0), x.stride(1), x.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        return out


def run(*args):
    return ModelNew()(*args)
