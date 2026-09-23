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
    BLOCK_F: tl.constexpr,
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
    var = acc_sumsq / F - mean * mean
    # Store two scalars: mean and sumsq/F (which is E[x^2])
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, single-element output tensor
    target_sparsity,      # float32 scalar
):
    # A&S approximation constants for inverse normal CDF
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
    q = 0.0
    # p is scalar target_sparsity
    p = target_sparsity
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        q = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6) / \
            ((((d1*z + d2)*z + d3)*z + d4)*z + 1.0)
    else:
        r = p - 0.5
        z2 = r * r
        q = (((((a1*z2 + a2)*z2 + a3)*z2 + a4)*z2 + a5)*z2 + a6) * r / \
            (((((b1*z2 + b2)*z2 + b3)*z2 + b4)*z2 + b5)*z2 + 1.0)
    # Store scalar invnorm
    tl.store(out_ptr, q)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32
    mean_ptr,             # *float32, mean per (b, s)
    sumsq_ptr,            # *float32, sum of squares per (b, s) divided by F
    inv_ptr,              # *float32, single-element invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input/output strides
    BLOCK_F: tl.constexpr,
):
    # Grid over (b, s, chunk of F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    base = b * stride_b + s * stride_s
    f_start = chunk * BLOCK_F

    offs = f_start + tl.arange(0, BLOCK_F)
    mask = offs < F

    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)

    # Load mean and std for (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)

    # Load invnorm scalar
    invnorm = tl.load(inv_ptr)

    threshold = mean + std * invnorm
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Computes mean and std per (b, s) using a Triton reduction kernel.
    - Computes invnorm(target_sparsity) via a Triton scalar kernel.
    - Applies ReLU(x - (mean + std * invnorm)) per element using a Triton elementwise kernel.
    - Returns output in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 and contiguous along the last dim for efficient Triton access
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq/F (E[x^2])
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
    grid_inv = (1,)  # single program
    invnorm_kernel[grid_inv](
        invnorm, target_sparsity,
        num_warps=1, num_stages=1
    )

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
        # Delegate to run(inputs, target_sparsity). The evaluator provides these.
        # Accept flexible args; if only one tensor is provided, default sparsity 0.01.
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)