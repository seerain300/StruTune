import torch
import triton
import triton.language as tl


@triton.jit
def mean_sumsq_kernel(
    x_ptr,               # *float32, input flattened or strided pointer (we pass contiguous float32)
    out_mean_ptr,        # *float32, shape [B*S]
    out_sumsq_ptr,       # *float32, shape [B*S]
    B, S, F,             # int sizes
    stride_b, stride_s, stride_f,  # int strides for x
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    # Scalar accumulators
    acc_sum = tl.zeros((), tl.float32)
    acc_sumsq = tl.zeros((), tl.float32)

    # Iterate over feature dimension in chunks
    BLOCK_F = 1024
    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance (unbiased=False)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(
    out_ptr,             # *float32, single-element output tensor
    target_sparsity,     # float32 scalar
):
    # Abramowitz & Stegun approximation for normal inverse CDF
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

    p = target_sparsity  # scalar
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Handle lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        if p > p_high:
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
def relu_threshold_kernel_1d(
    x_ptr, out_ptr, mean_ptr, sumsq_ptr, invnorm_ptr,
    B, S, F, stride_b, stride_s, stride_f, target_sparsity,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * S * F
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < total

    # Map linear index to (b, s, f)
    SF = S * F
    b = offs // SF
    rem = offs % SF
    s = rem // F
    f = rem % F

    base = b * stride_b + s * stride_s + f * stride_f

    x = tl.load(x_ptr + base, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    pid_bs = b * S + s
    mean = tl.load(mean_ptr + pid_bs)
    sumsq = tl.load(sumsq_ptr + pid_bs)
    std = tl.sqrt(sumsq)

    # Load invnorm scalar
    inv = tl.load(invnorm_ptr)  # single element

    cutoff = mean + std * inv
    y = x - cutoff
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + base, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
      threshold[b, s, :] = mean[b, s, :] + std[b, s, :] * invnormcdf(target_sparsity)
    Then applies ReLU(x - threshold) to create sparse activations.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for statistics; keep inputs as float32 (original did .to(torch.float32))
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq/F (std via sqrt(sumsq/F - mean^2))
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/sumsq reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_sumsq_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) via Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](float(target_sparsity), num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel over flattened space
    BLOCK = 1024  # chunk size for 1D grid
    total = B * S * F
    grid1 = (triton.cdiv(total, BLOCK),)
    relu_threshold_kernel_1d[grid1](
        x, out, mean, sumsq, invnorm, B, S, F, stride_b, stride_s, stride_f, float(target_sparsity),
        BLOCK=BLOCK, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
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
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)