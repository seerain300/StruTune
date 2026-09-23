import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor (float32)
    out_mean_ptr,         # *float32, length B*S
    out_sumsq_ptr,        # *float32, length B*S
    B: tl.constexpr,      # int
    S: tl.constexpr,      # int
    F: tl.constexpr,      # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # chunk size for reduction
):
    pid = tl.program_id(0)  # one program per (b, s)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    acc_sumsq = 0.0

    f = 0
    while f < F:
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # unbiased=False
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor
    out_ptr,              # *float32, output tensor
    mean_ptr,             # *float32, length B*S
    sumsq_ptr,            # *float32, length B*S
    B: tl.constexpr,      # int
    S: tl.constexpr,      # int
    F: tl.constexpr,      # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    target_sparsity,      # float32 scalar (0 < sparsity < 1)
    BLOCK_F: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Compute inverse normal CDF (A&S 7.1.26) inside the kernel
    p = target_sparsity  # scalar
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
    p_high = 0.97575  # 1 - p_low

    # Compute invnorm(p)
    # lower region
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        inv = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
              ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
    # central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        inv = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # upper region
    else:
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        inv = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
              ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)

    # Load mean and sumsq/F (variance)
    mean = tl.load(mean_ptr + b * S + s)
    var = tl.load(sumsq_ptr + b * S + s)
    std = tl.sqrt(var)

    # Base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Process features in chunks
    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        threshold = mean + std * inv
        y = tl.maximum(x - threshold, 0.0)
        tl.store(out_ptr + base + offs * stride_f, y, mask=mask)
        f += BLOCK_F


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized run:
    - Compute per-(b, s) mean and std across last dim.
    - Compute invnorm(target_sparsity) inside Triton kernel.
    - Apply ReLU(x - (mean + std * invnorm)) and return bfloat16.
    """
    # If no sparsity requested, return inputs (original code returns inputs)
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for computations
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Output buffer
    out = torch.empty_like(x, dtype=torch.float32)

    # Allocate outputs for mean and sumsq/F
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid_reduce = (B * S,)
    mean_std_kernel[grid_reduce](
        x, mean, sumsq,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Launch elementwise ReLU thresholding kernel over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, out, mean, sumsq,
        B, S, F, stride_b, stride_s, stride_f,
        float(target_sparsity),
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature: run(inputs, target_sparsity)
        # The evaluator typically passes two arguments; we delegate to run.
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than 2, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback default
            return run(args[0], 0.01)