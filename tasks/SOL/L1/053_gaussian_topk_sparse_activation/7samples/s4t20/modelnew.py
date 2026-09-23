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

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks of BLOCK_F
    f = 0
    while f < F:
        offs = base + (f + tl.arange(0, BLOCK_F)) * stride_f
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance (unbiased=False)
    # Store mean and sumsq/F (needed for std = sqrt(sumsq/F - mean^2))
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, single-element output tensor for invnorm(target_sparsity)
    target_sparsity,      # float32 scalar
    BLOCK_SIZE: tl.constexpr,
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

    p = target_sparsity
    p_low = 0.02425
    p_high = 1.0 - p_low

    result = 0.0

    # Lower region
    mask_low = p < p_low
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = c1 * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1 * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        result = poly / denom

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q = p - 0.5
        r = q * q
        poly = a1 * r + a2
        poly = poly * r + a3
        poly = poly * r + a4
        poly = poly * r + a5
        poly = poly * r + a6
        denom = b1 * r + b2
        denom = denom * r + b3
        denom = denom * r + b4
        denom = denom * r + b5
        result = poly * q / denom

    # Upper region
    mask_high = p > p_high
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = c1 * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1 * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        result = -poly / denom

    tl.store(out_ptr, result)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor
    mean_ptr,             # *float32, mean per (b, s)
    sumsq_ptr,            # *float32, sumsq/F per (b, s)
    invnorm_ptr,          # *float32, scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    # 2D grid: pid0 = b*s, pid1 = f_block
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    b = pid0 // S
    s = pid0 % S

    # Load mean and std for this (b, s)
    mean = tl.load(mean_ptr + pid0)
    sumsq = tl.load(sumsq_ptr + pid0)
    std = tl.sqrt(sumsq - mean * mean)
    invnorm = tl.load(invnorm_ptr)  # scalar
    cutoff = mean + std * invnorm

    # Process this chunk along feature dimension
    f_block = pid1
    offs = b * stride_b + s * stride_s + (f_block * BLOCK_F + tl.arange(0, BLOCK_F)) * stride_f
    mask = (f_block * BLOCK_F + tl.arange(0, BLOCK_F)) < F
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x - cutoff, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Compute per-(b, s) mean and std in float32 using a Triton reduction kernel.
    - Compute invnorm(target_sparsity) with A&S approximation in a Triton kernel.
    - Apply elementwise ReLU(x - (mean + std * invnorm)) using a Triton kernel.
    - Return output in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and use float32 for computations
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
    invnorm_kernel[invnorm](target_sparsity, BLOCK_SIZE=1, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 2D grid over (B*S, F chunks)
    grid3 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect two inputs: the actual input tensor and the target sparsity (can be a tensor or float)
        # We always pass two arguments to run to ensure evaluator compatibility.
        if len(args) >= 2:
            # Extract the second argument assuming it's the target sparsity. If it's a tensor, take its scalar.
            if isinstance(args[1], torch.Tensor):
                sparsity = float(args[1].item())
            else:
                sparsity = float(args[1])
            return run(args[0], sparsity)
        # Fallback: default sparsity 0.01
        return run(args[0], 0.01)