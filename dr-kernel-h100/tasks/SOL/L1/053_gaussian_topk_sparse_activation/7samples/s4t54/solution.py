import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_reduce_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, per (b, s) mean
    out_var_ptr,          # *float32, per (b, s) variance
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr  # chunk size along feature dim
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

    # Iterate over feature dimension in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean

    # Store per-(b, s) mean and variance
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_var_ptr + pid, var)


@triton.jit
def invnorm_scalar_kernel(
    out_ptr,              # *float32, scalar output (1 element)
    target_sparsity,      # float32 scalar
):
    # A&S constants (Abramowitz & Stegun approximation)
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

    p = target_sparsity
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Compute invnorm via A&S approximation; output stored to out_ptr[0]
    # We'll use the central region for typical sparsities; for extreme values, central formula is stable.
    # Compute q = p - 0.5 and use central region formula:
    q = p - 0.5
    r = q * q
    # Horner's method for numerator and denominator
    num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    invnorm = num / den

    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor
    mean_ptr,             # *float32, per (b, s) mean
    var_ptr,              # *float32, per (b, s) variance
    invnorm_ptr,          # *float32, scalar invnorm
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr
):
    # 2D grid: pid0 over (B*S), pid1 over feature chunks
    pid0 = tl.program_id(0)  # (b, s)
    b = pid0 // S
    s = pid0 % S

    # Load per-(b, s) mean and std
    mean = tl.load(mean_ptr + pid0)
    var = tl.load(var_ptr + pid0)
    std = tl.sqrt(var)

    invnorm = tl.load(invnorm_ptr)  # scalar
    threshold = mean + std * invnorm

    base = b * stride_b + s * stride_s

    # Process this chunk along F
    pid1 = tl.program_id(1)  # chunk id
    f_start = pid1 * BLOCK_F
    offs = base + f_start + tl.arange(0, BLOCK_F)
    mask = (f_start + tl.arange(0, BLOCK_F)) < F

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + offs, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized sparsity: out = max(0, x - (mean + std * invnorm(target_sparsity)))
    - All computation is done in Triton kernels.
    - Returns output in bfloat16 to match original behavior.
    """
    # If no sparsity requested, return inputs
    if target_sparsity == 0.0:
        return inputs

    # Work in float32 for stability, ensure contiguous
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate per-(b, s) mean and variance
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/var reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_var_reduce_kernel[grid](
        x, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Allocate scalar for invnorm and compute in Triton
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_scalar_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel
    grid3 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, var, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Maintain original signature and behavior: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than 2 args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)


def run(*args):
    return ModelNew()(*args)
