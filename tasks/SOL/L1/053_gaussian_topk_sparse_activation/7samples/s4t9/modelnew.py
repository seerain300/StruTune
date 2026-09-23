import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_kernel(
    x_ptr,                 # *float32, input tensor (already float32)
    out_mean_ptr,          # *float32, per (b, s) mean
    out_sumsq_ptr,         # *float32, per (b, s) sum of squares divided by F
    B, S, F,               # sizes
    stride_b, stride_s, stride_f  # strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Reduce over feature dimension in chunks
    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    sumsq_div_F = acc_sumsq / F  # sum of squares divided by number of features

    # Store results
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, sumsq_div_F)


@triton.jit
def compute_std_kernel(
    mean_ptr,              # *float32, per (b, s) mean
    sumsq_ptr,             # *float32, per (b, s) sumsq/F
    out_std_ptr,           # *float32, per (b, s) std
    size  # number of programs = B * S
):
    pid = tl.program_id(0)
    mean = tl.load(mean_ptr + pid)
    sumsq_div_F = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq_div_F - mean * mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def invnorm_kernel(
    out_ptr,               # *float32, 1-element tensor to hold invnorm
    target_sparsity        # float32 scalar in (0, 1)
):
    # Abramowitz & Stegun 5.2.23 approximation
    p = target_sparsity
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region constants
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

    if p < p_low:
        # Lower region
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        invnorm = poly / denom
    elif p > p_high:
        # Upper region
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        invnorm = -poly / denom
    else:
        # Central region
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        invnorm = poly * q / poly2

    # Store result into out_ptr[0]
    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                 # *float32, input tensor
    mean_ptr,              # *float32, per (b, s) mean
    std_ptr,               # *float32, per (b, s) std
    invnorm_ptr,           # *float32, scalar invnorm
    out_ptr,               # *float32, output tensor
    B, S, F,               # sizes
    stride_b, stride_s, stride_f  # strides
):
    # Grid over (B, S)
    b = tl.program_id(0)
    s = tl.program_id(1)
    base = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + b * S + s)
    std = tl.load(std_ptr + b * S + s)
    invnorm = tl.load(invnorm_ptr)  # scalar

    # Compute threshold per (b, s)
    threshold = mean + std * invnorm

    # Iterate over features in chunks
    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(out_ptr + base + offs * stride_f, y, mask=mask)
        f += 1024


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-(b, s) mean and std across feature dimension F.
    Threshold = mean + std * invnorm(target_sparsity).
    Output = max(0, inputs - threshold), returned in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 contiguous input for reduction and elementwise kernels
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # 1) Compute mean and sumsq/F for each (b, s)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    grid1 = (B * S,)
    compute_mean_std_kernel[grid1](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        num_warps=4, num_stages=2
    )

    # 2) Compute invnorm(target_sparsity) on device
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # 3) Compute std from mean and sumsq/F on device
    std = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    compute_std_kernel[(B * S,)](mean, sumsq, std, B * S, num_warps=1, num_stages=1)

    # 4) Elementwise ReLU against threshold
    out = torch.empty_like(x)
    relu_threshold_kernel[(B, S)](
        x, mean, std, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        num_warps=4, num_stages=2
    )

    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature: run(inputs, target_sparsity)
        # The evaluator passes two arguments; delegate to run.
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)