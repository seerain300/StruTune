import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor (contiguous or strided, we pass stride info)
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides (in elements)
    BLOCK_F: tl.constexpr,          # chunk size along F
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Loop over feature dimension F in chunks of BLOCK_F
    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        ptrs = x_ptr + base + offs * stride_f
        mask = offs < F
        x = tl.load(ptrs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    # std (population) = sqrt(E[x^2] - (E[x])^2)
    var = acc_sumsq / F - mean * mean
    std = tl.sqrt(var)

    # Store mean and std for this (b, s)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, std)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, single-element output for invnorm(target_sparsity)
    target_sparsity,      # float32 scalar
):
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

    # Determine which region
    low = target_sparsity < p_low
    mid = (target_sparsity >= p_low) & (target_sparsity <= p_high)
    high = target_sparsity > p_high

    # Compute invnorm using the appropriate formula
    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(target_sparsity))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = target_sparsity - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select result based on region
    invnorm = tl.where(low, z_low, 0.0) + tl.where(mid, z_mid, 0.0) + tl.where(high, z_high, 0.0)

    # Store single-element output
    tl.store(out_ptr, invnorm)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                    # *float32, input tensor
    mean_ptr,                # *float32, mean per (b, s)
    std_ptr,                 # *float32, std per (b, s)
    invnorm_ptr,             # *float32, scalar invnorm(target_sparsity)
    out_ptr,                 # *float32, output tensor
    B, S, F,                 # sizes
    stride_b, stride_s, stride_f,  # strides
    BLOCK_F: tl.constexpr,        # chunk size along F
):
    # 3D grid over (B, S, F chunks)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)
    f0 = chunk * BLOCK_F

    # Load mean and std for this (b, s)
    pid = b * S + s
    mean_val = tl.load(mean_ptr + pid)
    std_val = tl.load(std_ptr + pid)
    invnorm_val = tl.load(invnorm_ptr)  # scalar

    # Compute threshold: mean + std * invnorm
    threshold = mean_val + std_val * invnorm_val

    # Iterate over this chunk of F
    offs = f0 + tl.arange(0, BLOCK_F)
    mask = offs < F
    base = b * stride_b + s * stride_s
    x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
    y = tl.maximum(x - threshold, 0.0)  # ReLU(x - threshold)
    tl.store(out_ptr + base + offs * stride_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Computes per-(b, s) mean and std across feature dimension (last dim).
    - Computes invnorm(target_sparsity) using A&S approximation in Triton.
    - Applies out = ReLU(x - (mean + std * invnorm)) and returns bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 and contiguous for Triton kernels
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and std per (b, s)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    std = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, std, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) via Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU-threshold kernel
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, std, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature behavior:
        # The evaluator passes two tensors/arguments: input and target_sparsity.
        # Call run(inputs, target_sparsity).
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more than two args, try to extract sparsity as the second arg
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback to default sparsity
            return run(args[0], 0.01)