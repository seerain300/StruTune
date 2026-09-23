import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_var_ptr,          # *float32, output variance per (b, s)  [var = E[x^2] - mean^2]
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks of BLOCK_F
    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # unbiased=False (population variance)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_var_ptr + pid, var)


@triton.jit
def invnorm_kernel(
    out_ptr,  # *float32, single-element output tensor for invnorm(target_sparsity)
    target_sparsity,  # float32 scalar
):
    # Abramowitz & Stegun (26.2.23) approximation for inverse normal CDF
    # We compute the same constants as in the original _ndtri
    # Constants for lower region
    p_low = 0.02425
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

    # Constants for upper region
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

    # Single element output at out_ptr[0]
    p = target_sparsity  # scalar

    # Decide region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > (1.0 - p_low):
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(out_ptr, z)


@triton.jit
def relu_threshold_kernel(
    x_ptr, mean_ptr, var_ptr, invnorm_ptr, out_ptr,
    B, S, F,
    stride_b, stride_s, stride_f,
    target_sparsity,  # unused here, but kept for signature symmetry
    BLOCK_F: tl.constexpr,
):
    # 2D grid: (B*S, cdiv(F, BLOCK_F))
    pid_bs = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    b = pid_bs // S
    s = pid_bs % S

    base = b * stride_b + s * stride_s

    # Load mean and std (std = sqrt(var)) for this (b, s)
    mean = tl.load(mean_ptr + pid_bs)
    std = tl.sqrt(tl.load(var_ptr + pid_bs))

    # Load invnorm scalar (single element)
    inv = tl.load(invnorm_ptr)  # scalar

    # Compute threshold for this (b, s)
    threshold = mean + std * inv

    # Iterate over feature chunks
    f = pid_chunk * BLOCK_F
    offs = f + tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
      - Compute mean and variance (population) per (b, s) across feature dim.
      - Compute invnorm(target_sparsity) via A&S approximation in Triton.
      - Apply ReLU(x - (mean + std * invnorm)) elementwise.
    Returns output in bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and compute in float32
    x = inputs.contiguous()
    x_f32 = x.to(torch.float32)

    B, S, F = x_f32.shape
    stride_b, stride_s, stride_f = x_f32.stride()

    # Allocate outputs for mean and var per (b, s)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x_f32.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x_f32.device)

    # Launch mean/var reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x_f32, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) as a scalar in Triton
    invnorm = torch.empty((1,), dtype=torch.float32, device=x_f32.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer (float32)
    out = torch.empty_like(x_f32)

    # Elementwise ReLU-thresholding
    grid3 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x_f32, mean, var, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        target_sparsity,  # kept for signature symmetry
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature expectation: inputs tensor, target_sparsity float
        # The evaluator passes two arguments; delegate to run.
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