import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                 # *float32, input tensor
    out_mean_ptr,          # *float32, per (b, s) mean
    out_sumsq_ptr,         # *float32, per (b, s) sum(x^2)/F
    B, S, F,               # int sizes
    stride_b, stride_s, stride_f,  # input strides in elements
    BLOCK_F: tl.constexpr  # block size along feature dim
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
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    sumsq_over_F = acc_sumsq / F  # variance = sumsq_over_F - mean^2
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, sumsq_over_F)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                 # *float32, input tensor
    mean_ptr,              # *float32, per (b, s) mean
    sumsq_ptr,             # *float32, per (b, s) sumsq/F
    invnorm_scalar_ptr,    # *float32, scalar invnorm(target_sparsity)
    out_ptr,               # *float32, output tensor
    B, S, F,               # int sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # Grid is (B, S, cdiv(F, BLOCK_F))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)

    b = pid_b
    s = pid_s
    chunk = pid_c

    base = b * stride_b + s * stride_s

    # Compute (b, s) mean and std
    mean = tl.load(mean_ptr + b * S + s)
    sumsq = tl.load(sumsq_ptr + b * S + s)
    std = tl.sqrt(sumsq - mean * mean)

    # Load invnorm scalar
    inv = tl.load(invnorm_scalar_ptr)

    # Compute threshold = mean + std * inv
    threshold = mean + std * inv

    # Process a chunk of the feature dimension
    f_start = chunk * BLOCK_F
    offs = f_start + tl.arange(0, BLOCK_F)
    mask = offs < F

    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function) using A&S 7.1.26."""
    # Constants
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

    # Compute z = sqrt(-2*log(p)) for tails; central region uses p-0.5
    # We implement piecewise as in original code, but as host-side scalar logic.
    if p < p_low:
        q = torch.sqrt(torch.tensor(-2.0 * math.log(p), dtype=torch.float32, device=p.device))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        q = torch.sqrt(torch.tensor(-2.0 * math.log(1.0 - p), dtype=torch.float32, device=p.device))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    return float(z)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
    - Compute per-(b, s) mean and std across last dim (feature) in float32.
    - Compute invnorm(target_sparsity) on host (float32 scalar).
    - Apply ReLU(x - (mean + std * invnorm)) elementwise, return bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Convert to float32 and make contiguous for predictable strides
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sum(x^2)/F
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) as a scalar tensor on device
    invnorm_scalar = torch.tensor(_ndtri(float(target_sparsity)), dtype=torch.float32, device=x.device)

    # Output buffer in float32
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm_scalar, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Strictly mirror original signature: expect two args (inputs, target_sparsity)
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
