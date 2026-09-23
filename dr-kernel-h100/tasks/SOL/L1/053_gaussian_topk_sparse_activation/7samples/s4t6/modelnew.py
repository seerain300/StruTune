import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor (already float32)
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s) divided by F
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks of 1024
    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F  # sumsq/F, not yet subtracting mean^2
    # Store mean and sumsq/F (used later to compute std = sqrt(sumsq/F - mean^2))
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor (already float32)
    mean_ptr,             # *float32, per (b, s) mean
    sumsq_ptr,            # *float32, per (b, s) sumsq/F
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input/output strides
    invnorm_multiplier,   # float32 scalar: invnorm(target_sparsity)
):
    # Grid: (B, S, cdiv(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    cblk = tl.program_id(2)

    # Compute base pointers
    in_base = b * stride_b + s * stride_s
    out_base = b * stride_b + s * stride_s

    # Load mean and sumsq/F for this (b, s)
    mean = tl.load(mean_ptr + b * S + s)
    sumsq_over_F = tl.load(sumsq_ptr + b * S + s)
    std = tl.sqrt(sumsq_over_F - mean * mean)

    cutoff = mean + std * invnorm_multiplier

    # Process a chunk along feature dimension
    f_start = cblk * 1024
    offs = f_start + tl.arange(0, 1024)
    mask = offs < F
    x = tl.load(x_ptr + in_base + offs * stride_f, mask=mask, other=0.0)
    y = x - cutoff
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + out_base + offs * stride_f, y, mask=mask)


def _compute_invnorm(target_sparsity: float) -> float:
    """
    Inverse of the standard normal CDF using Abramowitz & Stegun 7.1.26 approximation.
    Matches the logic in the original _ndtri.
    """
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

    p = float(target_sparsity)
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = (-2.0 * tl.log(p)) ** 0.5
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = (-2.0 * tl.log(1.0 - p)) ** 0.5
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return float(z)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.

    Computes adaptive sparsity threshold per (batch, seq) position:
      mean = average across feature dim
      std  = population std across feature dim (unbiased=False)
      cutoff = mean + std * invnorm(target_sparsity)
    Then applies ReLU(x - cutoff). Returns output in bfloat16.
    """
    # Handle no sparsity
    if target_sparsity == 0.0:
        return inputs

    # Compute in float32 for stability
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

    # Compute invnorm(target_sparsity) as a Python float (host-side)
    invnorm = _compute_invnorm(target_sparsity)

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, out,
        B, S, F, stride_b, stride_s, stride_f, stride_b, stride_s, stride_f,  # note: output stride same as input
        invnorm,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Original signature expects (inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)