import torch
import triton
import triton.language as tl


@triton.jit
def relu_threshold_kernel_1d(
    x_ptr,               # *float32, input tensor as float32, shape [B, S, F] flattened
    mean_ptr,            # *float32, per-(b, s) mean, shape [B*S]
    std_ptr,             # *float32, per-(b, s) std, shape [B*S]
    out_ptr,             # *float32, output tensor as float32, shape [B, S, F] flattened
    B, S, F,             # int sizes
    stride_b, stride_s, stride_f,  # input strides for reconstructing (b, s, f)
    invnorm,             # float32 scalar: invnorm(target_sparsity)
):
    total = B * S * F
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < total

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute (b, s, f) from linear offsets
    # Note: idx = b*stride_b + s*stride_s + f*stride_f
    f_idx = offsets % stride_f
    tmp = offsets - f_idx
    s_idx = tmp % stride_s
    b_idx = tmp // stride_s

    # Compute per-(b, s) threshold: mean + std * invnorm
    mean_val = tl.load(mean_ptr + b_idx * S + s_idx, mask=mask, other=0.0)
    std_val = tl.load(std_ptr + b_idx * S + s_idx, mask=mask, other=0.0)
    threshold = mean_val + std_val * invnorm

    # Apply ReLU: max(0, x - threshold)
    y = tl.maximum(x - threshold, 0.0)

    tl.store(out_ptr + offsets, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation (Triton-optimized elementwise part).
    Computes per-(b, s) mean and std in PyTorch, then applies Triton kernel
    ReLU(x - (mean + std * invnorm(target_sparsity))) across features.
    Returns output in bfloat16 to match original behavior.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for numerical stability; keep contiguous layout
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Compute per-(b, s) mean and std along the last dim (features), unbiased=False
    mean = torch.mean(x, dim=-1)  # shape [B, S]
    std = torch.sqrt(torch.mean(x * x, dim=-1) - mean * mean)  # shape [B, S]

    # Compute invnorm(target_sparsity) using Abramowitz & Stegun approximation
    # Implemented as a scalar in host; could be a Triton kernel, but this is fast enough.
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

    # p = target_sparsity
    p = float(target_sparsity)

    if p < p_low:
        # Lower region
        q = torch.sqrt(-2.0 * torch.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        # Upper region
        q = torch.sqrt(-2.0 * torch.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # Central region
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    invnorm = float(z)

    # Allocate output buffer
    out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

    # Launch 1D Triton kernel over total elements
    grid = (triton.cdiv(B * S * F, 1024),)
    relu_threshold_kernel_1d[grid](
        x, mean.view(-1), std.view(-1), out,
        B, S, F, stride_b, stride_s, stride_f,
        invnorm,
        num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature and behavior: run(inputs, target_sparsity)
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