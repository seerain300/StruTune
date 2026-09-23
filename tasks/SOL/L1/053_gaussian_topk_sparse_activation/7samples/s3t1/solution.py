import torch
import torch.nn as nn
import math

import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(
    X_ptr,                  # *float32, input tensor pointer (flattened to 2D: [rows, F])
    mean_ptr,               # *float32, output mean per row pointer [rows]
    std_ptr,                # *float32, output std per row pointer [rows]
    F,                      # int: feature dimension length
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one [batch, seq] row)
    row_id = tl.program_id(0)
    # Accumulators for sum and sum of squares in float32
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over the feature dimension in tiles
    for offs in range(0, F, BLOCK_SIZE):
        col = offs + tl.arange(0, BLOCK_SIZE)
        mask = col < F
        x = tl.load(X_ptr + row_id * F + col, mask=mask, other=0.0)
        # x is float32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # Handle numerical issues: keep var >= 0, but original torch.std uses unbiased=False => var = E[x^2] - (E[x])^2
    # For typical data, var >= 0. If negative due to FP error, set to 0.
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def relu_threshold_kernel(
    X_ptr,               # *float32, input tensor pointer (flattened to 2D: [rows, F])
    C_ptr,               # *float32, cutoff per row pointer [rows]
    Y_ptr,               # *float32, output tensor pointer (flattened to 2D: [rows, F])
    F,                   # int: feature dimension length
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load cutoff for this row
    cutoff = tl.load(C_ptr + row_id)
    # Process the row in tiles
    for offs in range(0, F, BLOCK_SIZE):
        col = offs + tl.arange(0, BLOCK_SIZE)
        mask = col < F
        x = tl.load(X_ptr + row_id * F + col, mask=mask, other=0.0)  # float32
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + row_id * F + col, y, mask=mask)


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 7.1.26).
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

    result = torch.zeros_like(p)

    # Lower region
    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                           (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
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


class ModelNew(nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        - Compute per-row mean and std (float32) via Triton kernel.
        - Compute std_multiplier = inverse standard normal CDF(target_sparsity) on host (PyTorch).
        - Compute cutoff = mean + std * std_multiplier per row (float32).
        - Apply ReLU(input - cutoff) via Triton kernel, write float32 output.
        - Cast output to bfloat16 to match original behavior.
        """
        assert inputs.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, F = inputs.shape

        # Ensure contiguous and float32 for numerics
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Total number of rows = batch * seq
        rows = B * S

        # Allocate mean and std per row
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        # Launch Triton kernel to compute per-row mean and std
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x,
            mean,
            std,
            F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Compute std_multiplier (inverse CDF of target_sparsity) as a float32 scalar
        # Use the original approximation function but ensure scalar tensor on device
        std_multiplier = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=x.device))

        # Create cutoff per row: [rows] tensor
        # cutoff = mean + std * std_multiplier
        # std_multiplier is a 0-dim tensor; broadcast to [rows]
        cutoff = mean + std * std_multiplier

        # Allocate output for activation (float32)
        y = torch.empty_like(x, dtype=torch.float32)

        # Launch Triton kernel to apply ReLU(input - cutoff) elementwise per row
        grid_act = (rows,)
        relu_threshold_kernel[grid_act](
            x,
            cutoff,
            y,
            F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Cast to bfloat16 to match original output
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
