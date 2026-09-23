import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,                # *float32 input
    out_sum_ptr,          # *float32 output per (b, s)
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # strides
):
    pid = tl.program_id(0)  # one program per (b, s)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        f += 1024
    tl.store(out_sum_ptr + pid, acc_sum)


@triton.jit
def sumsq_kernel(
    x_ptr,                # *float32 input
    out_sumsq_ptr,        # *float32 output per (b, s)
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # strides
):
    pid = tl.program_id(0)  # one program per (b, s)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc_sumsq = 0.0
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024
    tl.store(out_sumsq_ptr + pid, acc_sumsq)


@triton.jit
def write_mean_std_kernel(
    out_mean_ptr,          # *float32, shape [B*S]
    out_std_ptr,           # *float32, shape [B*S]
    sum_ptr,               # *float32, shape [B*S]
    sumsq_ptr,             # *float32, shape [B*S]
    B, S, F,               # sizes
):
    # One program per (b, s) — here we just compute and write mean/std from sum/sumsq
    pid = tl.program_id(0)
    total = tl.load(sum_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    mean = total / F
    var = sumsq / F - mean * mean
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def invnorm_kernel(
    out_ptr,               # *float32, scalar output (1 element)
    target_sparsity,       # float32 scalar
    # A&S constants
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
    # Compute inverse normal CDF for p = target_sparsity
    p = target_sparsity  # scalar
    p_low = 0.02425
    p_high = 1.0 - p_low

    inv = 0.0
    # lower region
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        t = z
        # Horner's method for polynomial
        t = c1 * t + c2
        t = t * t + c3
        t = t * t + c4
        t = t * t + c5
        t = t * t + c6
        denom = d1 * t + d2
        denom = denom * t + d3
        denom = denom * t + d4
        denom = denom * t + 1.0
        inv = t / denom
    # central region
    elif p <= p_high:
        z = p - 0.5
        r = z * z
        t = r
        poly = a1 * t + a2
        poly = poly * t + a3
        poly = poly * t + a4
        poly = poly * t + a5
        poly = poly * t + a6
        t = r
        denom = b1 * t + b2
        denom = denom * t + b3
        denom = denom * t + b4
        denom = denom * t + b5
        denom = denom * t + 1.0
        inv = poly / denom
    # upper region
    else:
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        t = z
        t = c1 * t + c2
        t = t * t + c3
        t = t * t + c4
        t = t * t + c5
        t = t * t + c6
        denom = d1 * t + d2
        denom = denom * t + d3
        denom = denom * t + d4
        denom = denom * t + 1.0
        inv = -t / denom

    tl.store(out_ptr, inv)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32 input tensor
    out_ptr,              # *float32 output tensor
    mean_ptr,             # *float32, per (b, s) mean
    std_ptr,              # *float32, per (b, s) std
    invnorm_ptr,          # *float32, scalar invnorm
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # strides
):
    # Grid is (B, S, cdiv(F, BLOCK_F))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)

    b = pid_b
    s = pid_s
    base = b * stride_b + s * stride_s

    # Load per-(b, s) mean and std
    mean = tl.load(mean_ptr + (b * S + s))
    std = tl.load(std_ptr + (b * S + s))
    invnorm = tl.load(invnorm_ptr)  # scalar

    # Compute threshold scalar for this (b, s)
    threshold = mean + std * invnorm

    # Process feature chunks
    f = pid_c * 1024
    offs = base + f + tl.arange(0, 1024)
    mask = (f + tl.arange(0, 1024)) < F
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized implementation:
    - Compute mean and std per (b, s) using Triton reductions.
    - Compute invnorm using Triton.
    - Apply elementwise ReLU thresholding using Triton.
    Returns output in bfloat16, matching original.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 contiguous input
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for sums
    sum_per = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq_per = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reductions: one program per (b, s)
    grid = (B * S,)
    sum_kernel[grid](
        x, sum_per, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )
    sumsq_kernel[grid](
        x, sumsq_per, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Allocate mean and std per (b, s)
    mean_per = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    std_per = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Compute mean/std on device using PyTorch (tiny ops); or do in Triton via a tiny kernel
    # We'll use a small Triton kernel that reads sum/sumsq and writes mean/std.
    write_mean_std_kernel[grid](
        mean_per, std_per, sum_per, sumsq_per, B, S, F,
        num_warps=1, num_stages=1
    )

    # Compute invnorm(target_sparsity) in Triton (1-element output)
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](
        target_sparsity,
        num_warps=1, num_stages=1
    )

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, out, mean_per, std_per, invnorm, B, S, F, stride_b, stride_s, stride_f,
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
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)


def run(*args):
    return ModelNew()(*args)
