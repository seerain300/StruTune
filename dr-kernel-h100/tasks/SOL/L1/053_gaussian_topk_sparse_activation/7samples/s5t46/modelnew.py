import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_scalar_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation.
    # p_ptr: 1-element tensor (float32), z_ptr: 1-element tensor (float32)
    p = tl.load(p_ptr)  # scalar
    # Constants
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

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

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select piecewise
    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    tl.store(z_ptr, z_val)


@triton.jit
def row_reduce_sum_sumsq_kernel(X_ptr, Sum_ptr, Sumsq_ptr, H, BLOCK_SIZE: tl.constexpr, TILE_COUNT: tl.constexpr):
    # 1D grid over rows; each program reduces one row across H
    row = tl.program_id(0)  # pid along rows
    # Bounds check
    if row >= H:
        return

    # Row base offset (since we flattened to [S, H], row index is already the base offset)
    row_start = row * H  # but X_ptr is laid out contiguously; row base is row * H elements
    # Accumulators
    total = 0.0
    total_sq = 0.0

    # Loop over tiles
    for tile in range(TILE_COUNT):
        tile_start = tile * BLOCK_SIZE
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0)
        # Accumulate
        total += tl.sum(x)
        total_sq += tl.sum(x * x)

    # Write results
    tl.store(Sum_ptr + row, total)
    tl.store(Sumsq_ptr + row, total_sq)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, H):
    # One program per row; compute mean and std
    row = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row)
    sumsq_val = tl.load(Sumsq_ptr + row)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    # Ensure non-negative for numerical stability (should be exact if H >= 2)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, L, H, z_scalar_ptr, BLOCK_SIZE: tl.constexpr):
    # 2D grid: programs over rows and tiles of H
    row = tl.program_id(0)
    tile = tl.program_id(1)
    tile_start = tile * BLOCK_SIZE
    offs = tile_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load x for this row and tile
    x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)

    # Load per-row threshold (scalar)
    threshold = tl.load(Thresholds_ptr + row) + tl.load(z_scalar_ptr)  # threshold = mean + std * z_scalar

    # Gating: y = max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(Out_ptr + row * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    # Flatten [B, L, H] -> [S, H]
    B, L, H = inputs.shape
    S = B * L
    # We'll operate on float32; PyTorch original computes stats in float32 too
    X_flat = inputs.view(S, H).to(torch.float32)

    # 1) Compute z_scalar = _ndtri(target_sparsity) on device
    p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_scalar_kernel[(1,)](p_tensor, z_scalar)

    # 2) Row-wise sum and sum of squares (float32)
    Sum_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)

    BLOCK_SIZE = 1024
    TILE_COUNT = (H + BLOCK_SIZE - 1) // BLOCK_SIZE  # constexpr passed to kernel
    row_reduce_sum_sumsq_kernel[(S,)](X_flat, Sum_rows, Sumsq_rows, H, BLOCK_SIZE=BLOCK_SIZE, TILE_COUNT=TILE_COUNT, num_warps=4)

    # 3) Compute per-row mean and std
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    mean_std_kernel[(S,)](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, H, num_warps=4)

    # 4) Compute thresholds per row: threshold = mean + std * z_scalar
    thresholds = Mean_rows + Std_rows * z_scalar[0]  # elementwise broadcast along rows

    # 5) Elementwise gating: y = max(0, x - threshold)
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X_flat, thresholds, Out_flat, S, L, H, z_scalar, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out_flat.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        # Ensure CUDA tensor
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable