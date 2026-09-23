import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor (float32)
    out_mean_ptr,         # *float32, per-(b, s) mean
    out_std_ptr,          # *float32, per-(b, s) std
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def invnorm_scalar_kernel(  # Computes invnorm(target_sparsity) using A&S approximation
    out_ptr,             # *float32, single-element output
    target_sparsity,     # float32 scalar
):
    # A&S constants (Abramowitz & Stegun 7.1.26)
    p_low = 0.02426
    p_high = 1.0 - p_low

    # Coefficients for lower region
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    # Coefficients for upper region
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Central region coefficients (not needed for scalar invnorm)
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

    # Compute invnorm for target_sparsity
    # For scalar computation, use a single path. Since sparsity is between 0 and 1,
    # the central region always applies. We implement the central-region formula:
    # z = sqrt(2) * erfinv(sparsity - 0.5), but we avoid using torch.erfinv; instead
    # we use the A&S approximation for normal quantile.
    # We'll use the central-region formula with r = p - 0.5 and q = r^2.
    p = target_sparsity  # already in (0,1)
    r = p - 0.5
    q = r * r
    numerator = (((((a1 * q + a2) * q + a3) * q + a4) * q + a5) * q + a6) * r
    denominator = (((((b1 * q + b2) * q + b3) * q + b4) * q + b5) * q + 1.0)
    z = numerator / denominator  # invnorm(p)
    tl.store(out_ptr, z)


@triton.jit
def relu_threshold_kernel(
    x_ptr,        # *float32, input tensor
    mean_ptr,     # *float32, per-(b, s) mean
    std_ptr,      # *float32, per-(b, s) std
    inv_ptr,      # *float32, scalar invnorm(target_sparsity)
    out_ptr,      # *float32, output tensor
    B, S, F,      # int sizes
    stride_b_x, stride_s_x, stride_f_x,  # input strides
    stride_b_o, stride_s_o, stride_f_o,  # output strides
):
    # 2D grid: axis 0 over (B*S), axis 1 over chunks of F
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // S
    s = pid0 % S

    # Each program handles one chunk along F
    chunk = pid1
    f_start = chunk * 1024
    offs = b * stride_b_x + s * stride_s_x + f_start + tl.arange(0, 1024)
    mask = (f_start + tl.arange(0, 1024)) < F

    # Load mean and std for (b, s)
    mean = tl.load(mean_ptr + pid0)
    std = tl.load(std_ptr + pid0)

    # Load invnorm scalar
    invnorm = tl.load(inv_ptr)  # scalar

    # Compute threshold per element: mean + std * invnorm
    threshold = mean + std * invnorm

    # Load x, compute y = max(0, x - threshold)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x - threshold  # threshold is scalar
    y = tl.maximum(y, 0.0)  # ReLU

    # Store result
    out_offs = b * stride_b_o + s * stride_s_o + f_start + tl.arange(0, 1024)
    tl.store(out_ptr + out_offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized run:
    - Compute per-(b, s) mean and std over feature dimension.
    - Compute invnorm(target_sparsity) via Triton A&S approximation.
    - Apply ReLU(x - (mean + std * invnorm)) in Triton.
    Returns output in bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 contiguous for numerical stability
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b_x, stride_s_x, stride_f_x = x.stride()

    # Allocate per-(b, s) mean and std
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    std = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, std, B, S, F, stride_b_x, stride_s_x, stride_f_x,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm scalar via Triton
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_scalar_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer (float32)
    out = torch.empty_like(x, dtype=torch.float32)

    # Launch elementwise ReLU-threshold kernel: 2D grid over (B*S, cdiv(F, 1024))
    grid2 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid2](
        x, mean, std, invnorm, out,
        B, S, F,
        stride_b_x, stride_s_x, stride_f_x,
        stride_b_o=0, stride_s_o=0, stride_f_o=1,  # out is contiguous, can infer from strides
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature and behavior: run(inputs, target_sparsity)
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
