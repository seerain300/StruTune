import torch
import torch.nn.functional as F
import math

import triton
import triton.language as tl


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

    result = torch.zeros_like(p)

    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid.any():
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    mask_high = p > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
        result[mask_high] = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return result


@triton.jit
def _compute_row_sparsity_kernel(input_ptr, output_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Each Triton program handles one row (one (batch, seq) index).
    It:
      - reduces across the last dimension (N) to get mean and std of the row,
      - computes threshold = mean + std * std_multiplier,
      - applies output = max(0, input - threshold) for all elements in the row.
    All math in float32.
    """
    pid = tl.program_id(axis=0)
    # Number of rows = B * S; we assume input has been made contiguous with last dim as N.
    # For each row, base offset is pid * N (since the row length is N).
    base = pid * N

    # Accumulators in float32
    sum_x = 0.0
    sum_x2 = 0.0

    # First pass: compute sum and sum of squares
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Load a block of inputs as float32
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n_float = tl.float32(N)
    mean = sum_x / n_float
    # population variance (unbiased=False), then std
    var = sum_x2 / n_float - mean * mean
    # guard var from tiny negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: compute output = max(0, x - threshold)
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0)
        y = x - threshold
        # ReLU: y = max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(output_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # If no sparsity, return inputs unchanged (matches original behavior).
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            inputs = args[0][0]
        else:
            inputs = args[0]
        # If no sparsity, return directly
        if len(args) > 1:
            target_sparsity = args[1]
        else:
            target_sparsity = None

        if target_sparsity is None:
            # Assume default sparsity 0.1 if not provided; original code doesn't require second arg,
            # but here we mimic original signature. For safety, we require target_sparsity.
            raise ValueError("target_sparsity must be provided")

        # If target_sparsity == 0.0, return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure 3D input: [batch, seq, intermediate_size]
        # The original run function expects 1 input tensor; here we handle it generally.
        # If input is not 3D, we can try to flatten batch and seq dims. For safety, enforce 3D.
        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size]")

        B, S, N = inputs.shape

        # Compute std_multiplier on host using the same _ndtri approximation
        # _ndtri returns tensor of shape [1], take the scalar.
        std_multiplier = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)).item()

        # Ensure input is contiguous and float32 for kernel computation
        inputs_f32 = inputs.contiguous().to(torch.float32)

        # Allocate output in float32 (kernel computes and stores float32)
        output_f32 = torch.empty((B, S, N), dtype=torch.float32, device=inputs.device)

        # Launch Triton kernel: one program per row (B*S rows)
        grid = (B * S,)
        _compute_row_sparsity_kernel[grid](inputs_f32, output_f32, N, std_multiplier, BLOCK_SIZE=1024, num_warps=4)

        # Return in bfloat16 to match original behavior
        return output_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
