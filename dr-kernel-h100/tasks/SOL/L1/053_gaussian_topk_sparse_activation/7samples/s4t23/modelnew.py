import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base pointer for this (b, s) row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks of BLOCK_F
    for f in range(0, F, 1024):
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean
    # Store mean and var (std^2) per (b, s)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, var)


@triton.jit
def invnorm_kernel(
    out_ptr,  # *float32, single-element output tensor for invnorm(target_sparsity)
    target_sparsity,  # float32 scalar in (0, 1)
):
    # A&S approximation constants for inverse normal CDF
    p_low = 0.02425

    # Lower region approximation
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

    # Upper region approximation
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

    # Determine region
    if target_sparsity < p_low:
        # Lower region
        z = tl.sqrt(-2.0 * tl.log(target_sparsity))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        denom = (((((d1 * z + d2) * z + d3) * z + d4) * z) + 1.0)
        y = poly / denom
        invnorm = -((1.0 + y) * tl.sqrt(2.0 / math.pi) + (a1 * z + a2) * z * z * tl.exp(-0.5 * z * z))
    else:
        # Upper region
        t = 1.0 - target_sparsity
        z = tl.sqrt(-2.0 * tl.log(t))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        denom = (((((d1 * z + d2) * z + d3) * z + d4) * z) + 1.0)
        y = poly / denom
        invnorm = (1.0 - y) * tl.sqrt(2.0 / math.pi) + (a1 * z + a2) * z * z * tl.exp(-0.5 * z * z)

    # Store scalar result
    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr, mean_ptr, sumsq_ptr, invnorm_ptr, out_ptr,
    B, S, F,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # Grid: (B, S, cdiv(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    f = chunk * BLOCK_F
    offs = f + tl.arange(0, BLOCK_F)
    mask = offs < F

    # Base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Load input chunk
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    # Load mean and std per (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)

    # Load invnorm scalar (broadcast)
    invnorm = tl.load(invnorm_ptr)

    threshold = mean + std * invnorm
    y = x - threshold
    # ReLU: max(0, y)
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + base + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation of Gaussian-based top-k sparse activation.

    Computes adaptive sparsity threshold based on input statistics:
      threshold[b, s, :] = mean[b, s, :] + std[b, s, :] * norm.icdf(target_sparsity)
    Then applies ReLU(input - threshold) to create sparse activations.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity (all activations pass through).

    Returns:
        Sparsified tensor of same shape as input, returned in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for numerical stability; make contiguous
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq (std^2)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) in a Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](float(target_sparsity), num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
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
        # Match the original signature: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than 2, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback default
            return run(args[0], 0.01)