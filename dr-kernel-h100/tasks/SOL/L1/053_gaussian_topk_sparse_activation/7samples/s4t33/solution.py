import torch
import triton
import triton.language as tl


@triton.jit
def mean_sumsq_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    out_mean_ptr,         # *float32, output per (b, s): mean
    out_sumsq_ptr,        # *float32, output per (b, s): sumsq/F
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides for x
    BLOCK_F: tl.constexpr,           # chunk size along F
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

    # Iterate over F in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    sumsq_over_F = acc_sumsq / F
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, sumsq_over_F)


@triton.jit
def invnorm_a5223_kernel(
    out_ptr,              # *float32, single-element output
    p,                    # float32 scalar in (0, 1)
    # Abramowitz & Stegun 5.2.23 constants
    a1 = -3.969683028665376e+01,
    a2 = 2.209460984245205e+02,
    a3 = -2.759285104469687e+02,
    a4 = 1.383577518672690e+02,
    a5 = -3.066479806614716e+01,
    a6 = 2.506628277459239e+00,
    b1 = -5.447609879822406e+01,
    b2 = 1.615858368580409e+02,
    b3 = -1.556989798598866e+02,
    b4 = 6.680131188771972e+01,
    b5 = -1.328068155288572e+01,
    c1 = -7.784894002430293e-03,
    c2 = -3.223964580411365e-01,
    c3 = -2.400758277161838e+00,
    c4 = -2.549732539343734e+00,
    c5 = 4.374664141464968e+00,
    c6 = 2.938163982698783e+00,
    d1 = 7.784695709041462e-03,
    d2 = 3.224671290700398e-01,
    d3 = 2.445134137142996e+00,
    d4 = 3.754408661907416e+00,
):
    # Lower tail probability region
    p_low = 0.02425
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        poly2 = (((((d1 * z + d2) * z + d3) * z + d4) * z) + 1.0)
        y = poly / poly2
        invnorm = -y
    else:
        # Central region
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        y = (poly * q) / poly2
        # Upper region
        if p > (1.0 - p_low):
            z = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
            poly2 = (((((d1 * z + d2) * z + d3) * z + d4) * z) + 1.0)
            y2 = poly / poly2
            invnorm = y2
        else:
            invnorm = y
    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    mean_ptr,             # *float32, per (b, s) mean
    sumsq_ptr,            # *float32, per (b, s) sumsq/F
    inv_ptr,              # *float32, single-element invnorm
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input/output strides
    BLOCK_F: tl.constexpr,
):
    # Grid: (B, S, ceil(F / BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    c = tl.program_id(2)

    # Load mean and std for this (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq_over_F = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq_over_F)

    # Load invnorm scalar
    invnorm = tl.load(inv_ptr)  # shape [1] but scalar
    cutoff = mean + std * invnorm

    # Iterate over F chunk
    f_start = c * BLOCK_F
    offs = b * stride_b + s * stride_s + f_start + tl.arange(0, BLOCK_F)
    mask = (f_start + tl.arange(0, BLOCK_F)) < F
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x - cutoff
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + offs, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized implementation:
    - Compute per-(b, s) mean and std across feature dim (F).
    - Compute invnorm(target_sparsity) in Triton scalar kernel.
    - Apply elementwise ReLU(x - (mean + std * invnorm)).
    Returns output in bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for numerical stability in Triton kernels
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq/F
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq_over_F = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_sumsq_kernel[grid](
        x, mean, sumsq_over_F, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) in a Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_a5223_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x, dtype=torch.float32)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq_over_F, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than two args, try to extract sparsity as the second arg
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)


def run(*args):
    return ModelNew()(*args)
