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

    # Compute base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean
    # Store two scalars: mean and sumsq/F for std
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(
    out_ptr,  # *float32, single-element output tensor for invnorm(target_sparsity)
    target_sparsity,  # float32 scalar
):
    # A&S approximation constants
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

    # Compute based on region
    # Lower region
    q_low = torch.sqrt(-2.0 * torch.log(target_sparsity))
    r_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = target_sparsity - 0.5
    r_mid = (((((a1 * q_mid * q_mid + a2) * q_mid + a3) * q_mid + a4) * q_mid + a5) * q_mid + a6) * q_mid / \
            (((((b1 * q_mid * q_mid + b2) * q_mid + b3) * q_mid + b4) * q_mid + b5) * q_mid + 1.0)

    # Upper region
    q_up = torch.sqrt(-2.0 * torch.log(1.0 - target_sparsity))
    r_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select appropriate value
    # Use lower if p < p_low, upper if p > p_high, else mid
    # We can implement via masks and tl.where
    mask_low = target_sparsity < p_low
    mask_mid = (target_sparsity >= p_low) & (target_sparsity <= p_high)
    mask_up = target_sparsity > p_high

    result = tl.where(mask_low, r_low, 0.0)
    result = tl.where(mask_mid, r_mid, result)
    result = tl.where(mask_up, r_up, result)

    tl.store(out_ptr, result)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32
    mean_ptr,             # *float32, per-(b,s) mean
    sumsq_ptr,            # *float32, per-(b,s) sumsq/F
    invnorm_ptr,          # *float32, scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr,
):
    # Grid: (B, S, ceil_div(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    # Compute base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Load mean and std for this (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)
    # Load invnorm scalar
    invnorm = tl.load(invnorm_ptr)

    # Compute threshold = mean + std * invnorm
    threshold = mean + std * invnorm

    # Process a chunk of F
    f_start = chunk * BLOCK_F
    offs = base + f_start + tl.arange(0, BLOCK_F)
    mask = (f_start + tl.arange(0, BLOCK_F)) < F

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-(b, s) mean and std, then applies ReLU(x - (mean + std * invnorm(target_sparsity))).
    Returns output in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for numerical stability
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

    # Compute invnorm(target_sparsity) in a Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

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


def run(*args):
    return ModelNew()(*args)
