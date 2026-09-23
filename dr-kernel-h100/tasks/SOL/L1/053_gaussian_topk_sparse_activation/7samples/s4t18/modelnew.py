import torch
import triton
import triton.language as tl


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32
    mean_ptr,             # *float32, per (b, s) mean
    sumsq_ptr,            # *float32, per (b, s) sum of squares divided by F
    invnorm,              # float32 scalar computed on host
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides (in elements)
    BLOCK_F: tl.constexpr,
):
    # 3D grid: (B, S, cdiv(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Compute mean and std for this (b, s)
    # mean_ptr and sumsq_ptr are length B*S
    pid = b * S + s
    mean_val = tl.load(mean_ptr + pid)
    sumsq_val = tl.load(sumsq_ptr + pid)
    std_val = tl.sqrt(sumsq_val - mean_val * mean_val)

    # Threshold per (b, s): mean + std * invnorm
    threshold = mean_val + std_val * invnorm

    # Traverse feature dimension in chunks of BLOCK_F
    f_start = chunk * BLOCK_F
    while f_start < F:
        offs = base + f_start + tl.arange(0, BLOCK_F)
        mask = (f_start + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + offs, y, mask=mask)
        f_start += BLOCK_F


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s) divided by F
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides (in elements)
    BLOCK_F: tl.constexpr,
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
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, var)


def _ndtri_approx(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF using A&S approximation."""
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

    p = p
    p_low = 0.02425
    # Compute invnorm via A&S
    # Lower region
    q = torch.sqrt(-2.0 * torch.log(p))
    result_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Upper region
    q_up = torch.sqrt(-2.0 * torch.log(1.0 - p))
    result_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
                ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    result_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Combine
    # Note: p is a scalar here, so choose branch based on p
    # torch.where expects tensors of same shape; here scalar masks
    low_mask = p < p_low
    up_mask = p > (1.0 - p_low)
    result = torch.where(up_mask, result_up, torch.where(low_mask, result_low, result_mid))
    return result


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    - Compute per-(b, s) mean and std across feature dimension (last dim).
    - Compute cutoff = mean + std * invnorm(target_sparsity).
    - Output = ReLU(inputs - cutoff), returned as bfloat16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Work in float32 for stability, ensure contiguous
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Compute mean and std via Triton reduction
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) on host (single scalar)
    invnorm = _ndtri_approx(torch.tensor(target_sparsity, dtype=torch.float32, device=x.device))

    # Output buffer for elementwise kernel
    out = torch.empty_like(x, dtype=torch.float32)

    # Launch elementwise ReLU-threshold kernel
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm.item(), out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Cast to bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Delegate to run(inputs, target_sparsity) to match original Model behavior
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)