import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_reduce_kernel(
    x_ptr,                 # *float32, input tensor (contiguous), float32
    out_mean_ptr,          # *float32, output mean per (b, s)
    out_var_ptr,           # *float32, output variance per (b, s)
    B, S, F,               # int sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr, # block size along feature dim
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    # Accumulators in fp32
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
    var = acc_sumsq / F - mean * mean  # population variance (unbiased=False)

    # Store per-(b, s) mean and variance
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_var_ptr + pid, var)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                 # *float32, input tensor (contiguous), float32
    mean_ptr,              # *float32, mean per (b, s)
    var_ptr,               # *float32, var per (b, s)
    invnorm,               # float32 scalar, invnorm(target_sparsity)
    out_ptr,               # *float32, output tensor (contiguous), float32
    B, S, F,               # int sizes
    stride_b, stride_s, stride_f,  # input/output strides
    BLOCK_F: tl.constexpr, # block size along feature dim
):
    # 2D grid over (B*S, chunks of F)
    pid0 = tl.program_id(0)  # over (b, s)
    pid1 = tl.program_id(1)  # over F chunks

    b = pid0 // S
    s = pid0 % S

    base = b * stride_b + s * stride_s

    f_start = pid1 * BLOCK_F
    offs = f_start + tl.arange(0, BLOCK_F)
    mask = offs < F

    # Load input chunk
    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)

    # Load mean and std for (b, s)
    mean = tl.load(mean_ptr + pid0)
    std = tl.sqrt(tl.load(var_ptr + pid0))

    # Compute threshold = mean + std * invnorm
    threshold = mean + std * invnorm

    # Apply ReLU(x - threshold): max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store result
    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function) using A&S approximation."""
    # Abramowitz & Stegun approximation constants
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

    result = 0.0

    # Lower region
    if p < p_low:
        q = torch.sqrt(torch.tensor(-2.0 * torch.log(torch.tensor(p)), dtype=torch.float32)).item()
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    else:
        q = torch.sqrt(torch.tensor(-2.0 * torch.log(torch.tensor(1.0 - p)), dtype=torch.float32)).item()
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return float(result)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes per-(b, s) mean and std across feature dimension, then threshold
    threshold = mean + std * invnorm(target_sparsity)
    Applies ReLU(x - threshold). Returns in bfloat16.
    """
    # If no sparsity requested, just return inputs
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 computation
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate per-(b, s) mean and variance
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_var_reduce_kernel[grid](
        x, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) via host-side scalar function
    invnorm = _ndtri(float(target_sparsity))

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel: 2D grid over (B*S, F chunks)
    grid3 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, var, float(invnorm), out,
        B, S, F, stride_b, stride_s, stride_f, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Maintain original signature and behavior
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than 2 args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)