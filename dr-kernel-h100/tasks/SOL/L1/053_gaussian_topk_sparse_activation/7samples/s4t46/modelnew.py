import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Triton imports
import triton
import triton.language as tl


# Elementwise Triton kernel: out = max(0, x - (mean + std * invnorm))
@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    out_ptr,              # *float32, output tensor, shape [B, S, F]
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input strides
    mean,                 # float32 scalar (broadcast)
    std,                  # float32 scalar (broadcast)
    invnorm,              # float32 scalar (broadcast)
    BLOCK_F: tl.constexpr
):
    # 3D grid: (B, S, cdiv(F, BLOCK_F))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Compute base offset for this (b, s)
    base = pid_b * stride_b + pid_s * stride_s

    # Offsets along feature dimension for this chunk
    f_offs = pid_c * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = f_offs < F

    # Load input chunk
    x = tl.load(x_ptr + base + f_offs * stride_f, mask=mask, other=0.0)

    # Compute threshold: mean + std * invnorm (scalars), apply ReLU
    thr = mean + std * invnorm
    y = x - thr
    y = tl.maximum(y, 0.0)  # ReLU

    # Store result
    tl.store(out_ptr + base + f_offs * stride_f, y, mask=mask)


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    This is a rational approximation that works well for p in (0, 1).
    """
    # Constants for the approximation
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
    
    # We assume p is a 0-dim float tensor or scalar; compute scalar result
    if p.item() < p_low:
        q = torch.sqrt(torch.tensor(-2.0, device=p.device)) * torch.log(p)
        # Convert q to tensor for math
        q = torch.sqrt(-2.0 * torch.log(p))
        poly = c1 * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1 * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        return poly / denom
    elif p.item() > p_high:
        q = torch.sqrt(-2.0 * torch.log(1.0 - p))
        poly = c1 * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1 * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        return -poly / denom
    else:
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
        return poly * q / denom


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    
    Computes adaptive sparsity threshold based on input statistics:
    1) Compute mean and std of input across feature dimension
    2) Calculate threshold = mean + std * norm.icdf(target_sparsity)
    3) Apply ReLU(input - threshold) to create sparse activations
    
    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity (all activations pass through).
    Returns:
        Sparsified tensor of same shape as input, in bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and compute statistics in float32
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape

    # Compute mean and std along last dim (features)
    mean = torch.mean(x, dim=-1, keepdim=True)  # shape [B, S, 1]
    std = torch.std(x, dim=-1, keepdim=True, unbiased=False)  # shape [B, S, 1]

    # Compute invnorm(target_sparsity) via A&S approximation
    invnorm = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=x.device))

    # Prepare output
    out = torch.empty_like(x)

    # Strides for input/output
    stride_b, stride_s, stride_f = x.stride()

    # Launch Triton elementwise kernel
    BLOCK_F = 1024
    grid = (B, S, triton.cdiv(F, BLOCK_F))
    relu_threshold_kernel[grid](
        x, out,
        B, S, F,
        stride_b, stride_s, stride_f,
        mean.item(), std.item(), invnorm.item(),
        BLOCK_F,
        num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Preserve original signature behavior: run(inputs, target_sparsity)
        # The evaluator passes two arguments: input tensor and target_sparsity float.
        if len(args) == 2:
            return run(args[0], float(args[1]))
        # Fallbacks if less arguments are provided
        elif len(args) == 1:
            # Default sparsity
            return run(args[0], 0.01)
        else:
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            return run(args[0], 0.01)