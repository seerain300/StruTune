import torch
import torch.nn.functional as F
import math

# Triton is required for the kernel
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise y = max(0, x - threshold)
# Each program handles one (b, l) row and iterates across the last dimension (columns).
@triton.jit
def subtract_relu_kernel(
    input_ptr,          # *float32
    threshold_ptr,      # *float32, shape [B, L], but we use [B, L, 1] and index [b, l]
    output_ptr,         # *float32
    B: tl.constexpr,    # int
    L: tl.constexpr,    # int
    H: tl.constexpr,    # int (intermediate_size)
    stride_xb, stride_xl, stride_xh,   # strides for input
    stride_tb, stride_tl, stride_th,   # strides for threshold (we pass 3 dims but use 2)
    BLOCK_SIZE: tl.constexpr            # block size for iterating over H
):
    # program id: one per (b, l)
    pid = tl.program_id(0)
    b = pid // L
    l = pid % L

    # Load threshold scalar for this (b, l). threshold has shape [B, L, 1], so we index [b, l, 0].
    # Use provided strides for threshold.
    t_off = b * stride_tb + l * stride_tl  # + 0 * stride_th (third dim is 1)
    t = tl.load(threshold_ptr + t_off)

    # Iterate across the last dimension in blocks
    for j in range(0, H, BLOCK_SIZE):
        offs = j + tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Compute input offsets for [b, l, offs]
        x_offs = b * stride_xb + l * stride_xl + offs * stride_xh
        x = tl.load(input_ptr + x_offs, mask=mask, other=0.0)

        # Compute y = max(0, x - t)
        y = x - t
        y = tl.maximum(y, 0.0)

        # Store result
        out_offs = b * stride_xb + l * stride_xl + offs * stride_xh
        tl.store(output_ptr + out_offs, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes mean and std per (batch, seq) along last dim in PyTorch,
    then applies ReLU(inputs - adaptive_threshold) via Triton.
    Returns the sparsified tensor in bfloat16.
    """
    # Fallback to PyTorch if Triton/CUDA not available
    if (not TRITON_AVAILABLE) or (inputs.device.type != 'cuda'):
        # Original PyTorch path
        inputs_f32 = inputs.to(torch.float32)
        inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
        sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
        std_multiplier = _ndtri(sparsity_tensor)  # single scalar on device
        cutoff_threshold = inputs_mean + inputs_std * std_multiplier
        sparse_output = F.relu(inputs_f32 - cutoff_threshold)
        return sparse_output.to(torch.bfloat16)

    # Triton path: compute per-(b,l) mean/std in PyTorch, then sparsify in Triton
    inputs_f32 = inputs.to(torch.float32)
    # Compute mean and std along last dimension (keepdim=True for broadcasting)
    inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)

    # Compute std_multiplier = inverse CDF at target_sparsity (single scalar tensor)
    sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
    std_multiplier = _ndtri(sparsity_tensor)  # shape: [1] or scalar tensor

    # Form threshold tensor [B, L, 1] = mean + std * std_multiplier
    # Ensure contiguous for simple stride usage
    threshold = (inputs_mean + inputs_std * std_multiplier).to(torch.float32).contiguous()  # [B, L, 1]

    # Ensure inputs are contiguous along last dim
    x = inputs_f32.contiguous()

    B, L, H = x.shape
    # Allocate output
    output = torch.empty_like(x, dtype=torch.float32)

    # Strides
    stride_xb, stride_xl, stride_xh = x.stride()
    stride_tb, stride_tl, stride_th = threshold.stride()  # threshold is [B, L, 1]

    # Choose BLOCK_SIZE. 1024 works well for many cases; adjust if H is very large.
    BLOCK_SIZE = 1024

    # Launch one program per (b, l)
    grid = (B * L,)

    # Run Triton kernel
    subtract_relu_kernel[grid](
        x, threshold, output,
        B, L, H,
        stride_xb, stride_xl, stride_xh,
        stride_tb, stride_tl, stride_th,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # reasonable default; can tune
        num_stages=2  # reasonable default; can tune
    )

    # Cast back to bfloat16 (as original)
    return output.to(torch.bfloat16)


# Original helper: inverse standard normal CDF via Abramowitz-Stegun approximation
def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
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

    # We expect p to be a scalar tensor (0-dim or 1-dim length 1). Broadcast in PyTorch ops.
    result = torch.zeros_like(p)

    # Lower region
    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid.any():
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    mask_high = p > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
        result[mask_high] = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return result


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We assume the same input shape as the original Model: [batch_size, seq_len, intermediate_size]
        # If more than one argument is passed, we take the first one as the main input.
        if len(args) == 1:
            inputs = args[0]
        else:
            # In case the original Model.forward accepts multiple inputs, we just take the first.
            inputs = args[0]
        return run(inputs, target_sparsity=0.1)  # default sparsity; could be made configurable


def run(*args):
    return ModelNew()(*args)
