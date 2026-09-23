import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides (in elements)
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate across feature dimension in chunks of 1024
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # not used further, but computed for correctness
    # Store mean and sumsq/F (which equals E[x^2])
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, scalar output for invnorm(target_sparsity)
    target_sparsity,      # float32 scalar
):
    # A&S approximation for inverse normal CDF
    p = target_sparsity

    # Regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # Coefficients
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
        invnorm_val = poly / denom
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        c1 = -7.784894002430293e-03
        c2 = -3.223960984245205e-01
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
        invnorm_val = -poly / denom
    else:
        # Central region
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
        invnorm_val = poly * q / denom

    # Store scalar invnorm
    tl.store(out_ptr, invnorm_val)


@triton.jit
def relu_threshold_kernel(
    x_ptr,            # *float32 input
    mean_ptr,         # *float32, mean per (b, s)
    sumsq_ptr,        # *float32, sumsq/F per (b, s)
    invnorm_ptr,      # *float32 scalar invnorm
    out_ptr,          # *float32 output
    B, S, F,          # sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    # 3D grid: (B, S, chunks over F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    f_start = chunk * BLOCK_F
    offs = b * stride_b + s * stride_s + f_start + tl.arange(0, BLOCK_F)
    mask = (f_start + tl.arange(0, BLOCK_F)) < F

    # Load x
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load mean and std for (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)

    # Load invnorm scalar
    inv = tl.load(invnorm_ptr)

    # Compute threshold and apply ReLU(x - threshold)
    threshold = mean + std * inv
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store
    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original function:
      - Computes per-(b, s) mean and std across the last dimension (F).
      - Computes invnorm(target_sparsity).
      - Applies ReLU(x - (mean + std * invnorm)) elementwise.
    Returns output in bfloat16.
    """
    if target_sparsity == 0.0:
        # No sparsity, just return inputs in bfloat16
        return inputs.to(torch.bfloat16)

    # Work in float32 for stability
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq/F
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) via Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    grid_inv = (1,)
    invnorm_kernel[grid_inv](invnorm, float(target_sparsity), num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature and behavior: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)