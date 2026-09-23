import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_scalar_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation.
    # p_ptr: 1-element tensor (float32), z_ptr: 1-element tensor (float32)
    p = tl.load(p_ptr)
    # Constants
    p_low = 2.425e-2
    p_high = 1.0 - p_low
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
def reduce_sum_rows_kernel(X, Sum_rows, S, H, TILE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(axis=0)
    # Initialize sum accumulator
    acc = 0.0
    # Loop over tiles of H
    for tile in range(TILE):
        start = tile * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row start in flattened X
        row_start = row * H
        vals = tl.load(X + row_start + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    # Store per-row sum
    tl.store(Sum_rows + row, acc)


@triton.jit
def reduce_sumsq_rows_kernel(X, Sumsq_rows, S, H, TILE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    acc = 0.0
    for tile in range(TILE):
        start = tile * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        row_start = row * H
        vals = tl.load(X + row_start + offs, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_rows + row, acc)


@triton.jit
def compute_thresholds_kernel(Mean_rows, Std_rows, z_scalar, Thresholds, S):
    # Vectorized per-row computation: Thresholds[r] = Mean_rows[r] + Std_rows[r] * z_scalar
    r = tl.program_id(axis=0)
    mean = tl.load(Mean_rows + r)
    std = tl.load(Std_rows + r)
    z = tl.load(z_scalar)  # z_scalar is 1-element tensor
    threshold = mean + std * z
    tl.store(Thresholds + r, threshold)


@triton.jit
def gate_relu_kernel(X, Thresholds, Out, S, H, z_scalar, TILE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 2D grid: axis 0 = row, axis 1 = tile along H
    row = tl.program_id(axis=0)
    tile = tl.program_id(axis=1)
    start = tile * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    row_start = row * H
    x = tl.load(X + row_start + offs, mask=mask, other=0.0)
    thresh = tl.load(Thresholds + row)
    z = tl.load(z_scalar)  # scalar for broadcast check; not used here since we already have per-row thresh
    y = x - thresh
    y = tl.where(y > 0.0, y, 0.0)  # ReLU
    tl.store(Out + row_start + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dim, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    # Cast to float32 for numerical stability
    X = inputs.to(torch.float32)
    B, L, H = X.shape
    S = B * L

    # Flatten [B, L, H] -> [S, H]
    X_flat = X.view(S, H)

    # Allocate per-row reductions
    Sum_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=X.device)

    # Choose tile count and block size
    BLOCK_SIZE = 1024
    TILE = (H + BLOCK_SIZE - 1) // BLOCK_SIZE  # compile-time constexpr for Triton

    # Launch reduction kernels: one program per row
    reduce_sum_rows_kernel[(S,)](X_flat, Sum_rows, S, H, TILE=TILE, BLOCK_SIZE=BLOCK_SIZE)
    reduce_sumsq_rows_kernel[(S,)](X_flat, Sumsq_rows, S, H, TILE=TILE, BLOCK_SIZE=BLOCK_SIZE)

    # Compute per-row mean and std (population std, unbiased=False)
    mean = Sum_rows / H
    var = Sumsq_rows / H - mean * mean
    # Ensure non-negative variances due to numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute inverse normal CDF scalar for target_sparsity
    p_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    p_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=X.device)
    _ndtri_scalar_kernel[(1,)](p_tensor, z_scalar)

    # Compute per-row thresholds: threshold = mean + std * z_scalar
    Thresholds = torch.empty(S, dtype=torch.float32, device=X.device)
    compute_thresholds_kernel[(S,)](mean, std, z_scalar, Thresholds, S)

    # Allocate output and run elementwise gating
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=X.device)
    grid_gate = (S, TILE)
    gate_relu_kernel[grid_gate](X_flat, Thresholds, Out_flat, S, H, z_scalar[0], TILE=TILE, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out_flat.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable


def run(*args):
    return ModelNew()(*args)
