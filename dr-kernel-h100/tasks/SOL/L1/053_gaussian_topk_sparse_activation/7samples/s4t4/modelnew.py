import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor pointer
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides in elements
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    acc_sumsq = 0.0

    f = 0
    while f < F:
        offs = f + tl.arange(0, 1024)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, var)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, single-element output tensor for invnorm(target_sparsity)
    target_sparsity,      # float32 scalar
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
    p = target_sparsity

    # Choose region
    use_low = p < p_low
    use_high = p > (1.0 - p_low)

    # Compute invnorm in the central region by default; handle low/high explicitly
    q = 0.0
    r = 0.0
    invnorm = 0.0

    # Low region
    if use_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly_c = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly_d = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        invnorm = poly_c / poly_d
    else:
        # Central region
        q = p - 0.5
        r = q * q
        poly_a = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly_b = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        invnorm = (poly_a * q) / poly_b
        # High region
        if use_high:
            invnorm = -invnorm

    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor pointer
    mean_ptr,             # *float32, mean per (b, s)
    var_ptr,              # *float32, var per (b, s)
    invnorm_ptr,          # *float32, scalar invnorm
    out_ptr,              # *float32, output tensor pointer
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input/output strides
    target_sparsity,      # scalar (not used in kernel, stored invnorm)
    BLOCK_F: tl.constexpr
):
    # 2D grid: (B*S, cdiv(F, BLOCK_F))
    pid_bs = tl.program_id(0)
    pid_blk = tl.program_id(1)
    b = pid_bs // S
    s = pid_bs % S

    base = b * stride_b + s * stride_s
    start = pid_blk * BLOCK_F

    offs = start + tl.arange(0, BLOCK_F)
    mask = offs < F

    # Load x
    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    mean = tl.load(mean_ptr + pid_bs)
    std = tl.sqrt(tl.load(var_ptr + pid_bs))

    # Load invnorm scalar
    invnorm = tl.load(invnorm_ptr)

    # Threshold
    threshold = mean + std * invnorm

    # Elementwise ReLU: max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes per-(b, s) mean and std across feature dim, then applies:
      out = max(0, x - (mean + std * invnorm(target_sparsity)))
    Returns bfloat16 tensor.
    """
    if target_sparsity == 0.0:
        return inputs

    # Work in float32 for numerical stability
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and var (sum of squares / F)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/var reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) using Triton
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, chunks over F)
    grid3 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, var, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        target_sparsity,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature and behavior: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)