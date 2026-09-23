import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl


@triton.jit
def _relu_adaptive_threshold_kernel(
    x_ptr,            # *fp32, flattened input
    thresholds_ptr,   # *fp32, 1D thresholds of length rows
    out_ptr,          # *fp32, flattened output
    N,                # int32, total number of elements
    rows,             # int32, number of rows = batch_size * seq_len
    feat,             # int32, number of features per row = intermediate_size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute row index for each element: rs = (offset // feat)
    rs = offsets // feat

    # Load threshold for each row
    thr = tl.load(thresholds_ptr + rs, mask=mask, other=0.0)

    # Apply adaptive ReLU: y = max(0, x - thr[rs])
    y = x - thr
    y = tl.maximum(y, 0.0)

    # Store results
    tl.store(out_ptr + offsets, y, mask=mask)


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

    # Compute in float32
    p32 = p.to(torch.float32)
    result = torch.zeros_like(p32)

    mask_low = p32 < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p32[mask_low]))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result[mask_low] = poly / den

    mask_mid = (p32 >= p_low) & (p32 <= p_high)
    if mask_mid.any():
        q = p32[mask_mid] - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        result[mask_mid] = num / den

    mask_high = p32 > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p32[mask_high]))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result[mask_high] = -poly / den

    return result


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA tensors for Triton; cast to float32 for numerics
        device = inputs.device
        inputs_f32 = inputs.to(torch.float32)

        # Compute per-row statistics along last dimension
        # inputs_f32 shape: [B, S, F]
        B, S, F = inputs_f32.shape
        # Mean and std along last dim (feature), keepdim=True to get [B, S, 1]
        inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)

        # Compute std multiplier from target_sparsity: inverse normal CDF
        # This is a scalar on the same device
        std_multiplier = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=device))

        # Compute adaptive cutoff threshold per row
        cutoff_threshold = inputs_mean + inputs_std * std_multiplier  # shape [B, S, 1]
        # Flatten thresholds to 1D for easy indexing by row index in kernel
        rows = B * S
        thresholds = cutoff_threshold.view(rows).contiguous()

        # Flatten input and prepare output
        x_flat = inputs_f32.view(-1).contiguous()
        N = x_flat.numel()
        out_flat = torch.empty(N, device=device, dtype=torch.float32)

        # Launch Triton kernel
        BLOCK_SIZE = 4096  # tuneable; 1024-8192 are typical good choices
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _relu_adaptive_threshold_kernel[grid](
            x_flat, thresholds, out_flat, N, rows, F,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # tuneable
        )

        # Reshape output to original shape
        out = out_flat.view(B, S, F)
        return out