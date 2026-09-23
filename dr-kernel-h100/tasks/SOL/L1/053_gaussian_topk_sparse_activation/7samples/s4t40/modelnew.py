import torch
import triton
import triton.language as tl


@triton.jit
def mean_sumsq_kernel(
    x_ptr,                 # *float32, input tensor as float32
    out_mean_ptr,          # *float32, output mean per (b, s)
    out_sumsq_ptr,         # *float32, output sum of squares per (b, s)
    B, S, F,               # sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr, # chunk size along feature dim
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base pointer for this (b, s)
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
    # population variance (unbiased=False): var = E[x^2] - (E[x])^2
    var = acc_sumsq / F - mean * mean
    # store mean and std
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, tl.sqrt(var))


@triton.jit
def relu_threshold_kernel(
    x_ptr,                 # *float32, input tensor
    mean_ptr,              # *float32, per-(b, s) mean
    std_ptr,               # *float32, per-(b, s) std
    out_ptr,               # *float32, output tensor
    B, S, F,               # sizes
    stride_b, stride_s, stride_f,  # input/output strides
    invnorm,               # scalar float32: invnormcdf(target_sparsity)
    BLOCK_F: tl.constexpr,
):
    # 3D grid: (b, s, chunk along F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    base = b * stride_b + s * stride_s
    f_off = chunk * BLOCK_F

    offs = base + f_off + tl.arange(0, BLOCK_F)
    mask = (f_off + tl.arange(0, BLOCK_F)) < F

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)

    # Adaptive cutoff: mean + std * invnorm
    cutoff = mean + std * invnorm

    # Elementwise ReLU: max(0, x - cutoff)
    y = x - cutoff
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offs, y, mask=mask)


def _ndtri(p: float) -> float:
    """Abramowitz and Stegun 7.1.26 approximation for inverse normal CDF."""
    # Constants for the approximation
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

    # Compute in float32
    p = float(p)
    if p < p_low:
        q = (p > 0.0) * (p < 1.0) * (-2.0 * math.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = (p - 0.5)
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = (p > 0.0) * (p < 1.0) * (-2.0 * math.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return float(result)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    - Computes per-(b, s) mean and std across feature dim (last dim).
    - Computes cutoff = mean + std * invnormcdf(target_sparsity).
    - Applies ReLU(input - cutoff) elementwise.
    Returns output in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 and contiguous for Triton
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and std (per (b, s))
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    std = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_sumsq_kernel[grid](
        x, mean, std, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) on host (scalar), matching original approximation
    invnorm = float(_ndtri(target_sparsity))

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, std, out, B, S, F, stride_b, stride_s, stride_f,
        invnorm, BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature: run(inputs, target_sparsity)
        # The evaluator passes two arguments; we delegate to run.
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