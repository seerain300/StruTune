import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides (in elements)
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

    # Iterate over feature dimension F in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # Reduce this chunk into scalars
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
    out_ptr,              # *float32, 1-element output tensor for invnorm(target_sparsity)
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

    # p_low and high setup
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Scalar computation for invnorm
    # Central region
    q = target_sparsity - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    invnorm_center = poly * q / poly2

    # Masks
    mask_low = target_sparsity < p_low
    mask_high = target_sparsity > p_high

    # Lower region
    q_low = torch.sqrt(torch.tensor(-2.0, device=target_sparsity.device)) * torch.log(torch.tensor(p_low, device=target_sparsity.device))
    approx_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                 ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    invnorm_low = approx_low

    # Upper region
    q_high = torch.sqrt(torch.tensor(-2.0, device=target_sparsity.device)) * torch.log(torch.tensor(1.0 - p_high, device=target_sparsity.device))
    approx_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                  ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    invnorm_high = approx_high

    # Select final invnorm value
    invnorm_val = torch.where(mask_low, invnorm_low, torch.where(mask_high, invnorm_high, invnorm_center))
    # Store as float32 into out_ptr[0]
    tl.store(out_ptr, invnorm_val)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32
    mean_ptr,             # *float32, mean per (b, s)
    sumsq_ptr,            # *float32, sumsq/F per (b, s)
    invnorm_ptr,          # *float32, 1-element tensor with invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    # Grid: (B, S, ceil_div(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    f_start = chunk * BLOCK_F
    offs_f = tl.arange(0, BLOCK_F) + f_start
    mask = offs_f < F

    # Base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Load input chunk
    x = tl.load(x_ptr + base + offs_f, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)

    # Load invnorm scalar
    invnorm = tl.load(invnorm_ptr)

    # Compute threshold and apply ReLU
    threshold = mean + std * invnorm
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store result
    tl.store(out_ptr + base + offs_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
      - mean and std across feature dimension (last dim)
      - threshold = mean + std * invnormcdf(target_sparsity)
    Applies ReLU(input - threshold). Returns in bfloat16.
    """
    # No sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Work in float32 for stability
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # 1) Compute mean and sumsq/F per (b, s) using Triton reduction kernel
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # 2) Compute invnorm(target_sparsity) using Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[(1,)](invnorm, float(target_sparsity), num_warps=1, num_stages=1)

    # 3) Elementwise ReLU with threshold in Triton
    out = torch.empty_like(x)
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
        # Delegate to run(inputs, target_sparsity) as in the original Model
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)