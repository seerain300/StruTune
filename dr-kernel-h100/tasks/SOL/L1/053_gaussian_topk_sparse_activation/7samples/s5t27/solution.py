import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_scalar_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation.
    # Writes result to z_ptr[0].
    p = tl.load(p_ptr)
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

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

    # Compute q for lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
              ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select piecewise
    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    tl.store(z_ptr, z_val)


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S, L, H, stride_x, BLOCK_SIZE: tl.constexpr):
    # One program per (b, l) row: compute sum across H
    b = tl.program_id(0)
    l = tl.program_id(1)
    row_start = b * L + l
    # Accumulator
    acc = 0.0
    # Iterate over H in tiles
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        # Row-major addressing: element at (b, l, idx) has linear offset row_start * stride_x + idx
        x = tl.load(X_ptr + row_start * stride_x + idx, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    # Store sum for this row
    tl.store(Sum_ptr + b * L + l, acc)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, S, L, H, stride_x, BLOCK_SIZE: tl.constexpr):
    # One program per (b, l) row: compute sum of squares across H
    b = tl.program_id(0)
    l = tl.program_id(1)
    row_start = b * L + l
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(X_ptr + row_start * stride_x + idx, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    tl.store(Sumsq_ptr + b * L + l, acc)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, L, H, stride_x, BLOCK_SIZE: tl.constexpr):
    # 2D grid: (B, L) over rows, and tiles over H
    b = tl.program_id(0)
    l = tl.program_id(1)
    row_start = b * L + l
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        # Load input row tile and corresponding threshold (scalar per row)
        x = tl.load(X_ptr + row_start * stride_x + idx, mask=mask, other=0.0)
        thresh = tl.load(Thresholds_ptr + b * L + l)
        y = x - thresh
        # ReLU
        y = tl.where(y > 0, y, 0.0)
        tl.store(Out_ptr + row_start * stride_x + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-(b, l) mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Shape
    B, L, H = inputs.shape
    # Flatten to [S, H] logical view but use 2D grid over (B, L) for per-row stats
    # We will launch row_sum and row_sumsq with grid (B, L), and gate with grid (B, L, tiles).

    # Cast input to float32 for stable math
    inputs_f32 = inputs.to(torch.float32)

    # Stride along last dim (contiguous): stride_x = H
    stride_x = H

    # Allocate buffers for sums
    Sum = torch.empty(B * L, dtype=torch.float32, device=inputs.device)
    Sumsq = torch.empty(B * L, dtype=torch.float32, device=inputs.device)

    # Launch reductions: one program per (b, l)
    BLOCK_SIZE = 1024
    grid_reduce = (B, L)
    row_sum_kernel[grid_reduce](inputs_f32, Sum, B, L, H, stride_x, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](inputs_f32, Sumsq, B, L, H, stride_x, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row (mean = sum/H, var = sumsq/H - mean^2)
    # Use vector arithmetic on device
    mean_rows = Sum / float(H)
    var_rows = (Sumsq / float(H)) - (mean_rows * mean_rows)
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var_rows = tl.maximum(var_rows, 0.0)
    std_rows = tl.sqrt(var_rows)

    # Compute z_scalar = _ndtri(target_sparsity) on device via Triton kernel
    p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p_tensor[0] = float(target_sparsity)
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_scalar_kernel[(1,)](p_tensor, z_tensor)  # single program handles scalar

    # Precompute thresholds per (b, l) row: threshold = mean + std * z_scalar
    thresholds = mean_rows + std_rows * z_tensor[0]

    # Allocate output for gating
    Out = torch.empty_like(inputs_f32, dtype=torch.float32, device=inputs.device)

    # Launch gating: 2D grid over (B, L) and H tiles
    grid_gate = (B, L)
    gate_relu_kernel[grid_gate](inputs_f32, thresholds, Out, B, L, H, stride_x, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    return Out.view(B, L, H).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        # Run Triton-only forward
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable


def run(*args):
    return ModelNew()(*args)
