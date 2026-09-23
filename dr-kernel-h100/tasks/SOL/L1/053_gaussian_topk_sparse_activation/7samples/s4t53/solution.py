import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_reduce_kernel(
    x_ptr,                # *float32, input as float32
    out_mean_ptr,         # *float32, per (b, s) mean
    out_var_ptr,          # *float32, per (b, s) variance
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr  # chunk size along feature dim
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base pointer offset for this (b, s)
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
    var = acc_sumsq / F - mean * mean
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_var_ptr + pid, var)


@triton.jit
def invnorm_scalar_kernel(
    out_ptr,              # *float32, single-element output tensor
    target_sparsity       # float32 scalar
):
    # A&S constants for inverse normal CDF (good up to ~0.9999)
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

    # p < 0.5: lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    r = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # p >= 0.5: upper region
    q2 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    r2 = -(((((c1 * q2 + c2) * q2 + c3) * q2 + c4) * q2 + c5) * q2 + c6) / \
         ((((d1 * q2 + d2) * q2 + d3) * q2 + d4) * q2 + 1.0)

    invnorm = tl.where(p < 0.5, r, r2)

    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input as float32
    mean_ptr,             # *float32, per (b, s) mean
    var_ptr,              # *float32, per (b, s) var
    invnorm,              # float32 scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # 3D grid: (B, S, chunks over F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    base = b * stride_b + s * stride_s
    f = chunk * BLOCK_F

    offs = f + tl.arange(0, BLOCK_F)
    mask = offs < F

    # Load input chunk
    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)

    # Load mean and std for this (b, s)
    mean = tl.load(mean_ptr + b * S + s)
    std = tl.sqrt(tl.load(var_ptr + b * S + s))

    # Apply threshold and ReLU
    threshold = mean + std * invnorm
    y = x - threshold
    y = tl.where(y > 0.0, y, 0.0)

    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold per (batch, seq) based on input stats:
      threshold = mean + std * invnorm(target_sparsity)
    Then applies ReLU(x - threshold) elementwise.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level (0.0 means no sparsity).
    Returns:
        Output tensor in bfloat16, same shape as input.
    """
    if target_sparsity == 0.0:
        # No sparsity: return original inputs in bfloat16
        return inputs.to(torch.bfloat16)

    # Ensure contiguous and compute in float32
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and var (per (b, s))
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_var_reduce_kernel[grid](
        x, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) in Triton (scalar)
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_scalar_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer (float32 for computation)
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, var, invnorm[0], out,
        B, S, F, stride_b, stride_s, stride_f,
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


def run(*args):
    return ModelNew()(*args)
