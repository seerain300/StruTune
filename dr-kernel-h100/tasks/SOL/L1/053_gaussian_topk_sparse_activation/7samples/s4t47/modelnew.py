import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,            # *float32, input tensor
    out_ptr,          # *float32, output sum per (b, s)
    B, S, F,          # sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc = 0.0
    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
        f += 1024

    tl.store(out_ptr + pid, acc)


@triton.jit
def sumsq_kernel(
    x_ptr,            # *float32, input tensor
    out_ptr,          # *float32, output sum of squares per (b, s)
    B, S, F,          # sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc = 0.0
    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
        f += 1024

    tl.store(out_ptr + pid, acc)


@triton.jit
def relu_threshold_kernel(
    x_ptr,            # *float32, input tensor
    out_ptr,          # *float32, output tensor
    B, S, F,          # sizes
    stride_b, stride_s, stride_f,  # input strides
    mean_ptr,         # *float32, 1-element tensor with mean per (b, s)
    std_ptr,          # *float32, 1-element tensor with std per (b, s)
    invnorm,          # float32 scalar: invnorm(target_sparsity)
    BLOCK_F: tl.constexpr,
):
    # Grid over (B, S, chunks of F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    c = tl.program_id(2)

    start = c * BLOCK_F
    offs = start + tl.arange(0, BLOCK_F)
    mask = offs < F

    base = b * stride_b + s * stride_s

    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    mean = tl.load(mean_ptr + b * S + s)
    std = tl.load(std_ptr + b * S + s)

    threshold = mean + std * invnorm
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function) using A&S approximation.

    Matches the behavior in the original _ndtri helper.
    """
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

    # Fast branches using approximation
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return poly / den
    elif p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return -poly / den
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return poly * q / den


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation with Triton kernels.

    1) Compute per-(batch, seq) mean and std across feature dimension.
    2) Compute invnorm(target_sparsity) using A&S approximation.
    3) Apply ReLU(x - (mean + std * invnorm)) elementwise via Triton.

    Returns bfloat16 tensor.
    """
    # Ensure device is CUDA and contiguous
    assert inputs.is_cuda, "Inputs must be on CUDA device for Triton."
    x = inputs.contiguous().to(torch.float32)
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for sum and sum of squares per (b, s)
    sum_all = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq_all = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch sum reduction kernel: one program per (b, s)
    grid = (B * S,)
    sum_kernel[grid](x, sum_all, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=1024, num_warps=4, num_stages=2)

    # Launch sumsq reduction kernel: one program per (b, s)
    sumsq_kernel[grid](x, sumsq_all, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=1024, num_warps=4, num_stages=2)

    # Compute mean and std on device (elementwise ops)
    mean = sum_all / float(F)
    # population std: E[x^2] - (E[x])^2
    var = (sumsq_all / float(F)) - mean * mean
    std = torch.sqrt(var)

    # Compute invnorm(target_sparsity) using A&S approximation (single scalar)
    invnorm = float(_ndtri(target_sparsity))

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise Triton kernel: grid over (B, S, chunks of F)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, out, B, S, F, stride_b, stride_s, stride_f,
        mean, std, invnorm,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Delegate to run(inputs, target_sparsity), matching original signature
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)