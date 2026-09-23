import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S
    base = b * S + s
    base_idx = base * D  # scalar int32 base offset

    sum_val = 0.0
    sumsq_val = 0.0

    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


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
    P_ptr,           # *float32, length 1 (scalar sparsity p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p (single element)
    p = tl.load(P_ptr)
    # Abramowitz & Stegun 5.2.23 approximation
    # For p in (0,1), use piecewise approximation
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # Rational approximation coefficients
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
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        denom = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        out = poly / denom
    elif p <= p_high:
        q = p - 0.5
        r = q * q
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
        poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        denom = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        out = poly * q / denom
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
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
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        denom = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        out = -poly / denom

    tl.store(OUT_ptr, out)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D] (we'll cast to bfloat16 on host)
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)
    base = b * S + s
    base_idx = base * D

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold  # broadcast scalar
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are contiguous and on device
        x = inputs.contiguous()
        B, S, D = x.shape

        # Device-side buffers for sums, sumsq, mean, std (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Device-side scalar for sparsity (float32), 1-element tensor for z-score
        p_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        p_buf[0] = float(target_sparsity)  # single element on device
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per (b, s)
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            x, sum_buf, sumsq_buf,
            B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # Launch mean/std kernel: one program per (b, s)
        grid_meanstd = (B * S,)
        compute_mean_std_kernel[grid_meanstd](
            sum_buf, sumsq_buf,
            mean_buf, std_buf,
            D,
            num_warps=1,
        )

        # Launch ndtri approx kernel: single program
        grid_ndtri = (1,)
        ndtri_approx_kernel[grid_ndtri](
            p_buf, z_buf,
            num_warps=1,
        )

        # Allocate output in float32 for kernel store; we’ll cast to bfloat16 after
        out_f32 = torch.empty(B, S, D, dtype=torch.float32, device=x.device)

        # Launch activation kernel: 3D grid over (b, s, tiles of D)
        grid_apply = (B, S, (D + 1023) // 1024)
        apply_activation_kernel[grid_apply](
            x, mean_buf, std_buf, z_buf, out_f32,
            B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # Match original behavior: return bfloat16
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
