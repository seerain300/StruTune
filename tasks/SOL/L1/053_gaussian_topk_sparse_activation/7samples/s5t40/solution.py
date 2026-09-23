import torch
import triton
import triton.language as tl


# Triton kernels

@triton.jit
def sum_rows_kernel(X_ptr, Sum_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row sum across H. Grid is (S, tiles), each program handles one tile of size BLOCK_SIZE
    and atomically adds to Sum_ptr[row].
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    # X_ptr layout is row-major: offset = row * H + col
    x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)  # sum across the vector
    tl.atomic_add(Sum_ptr + row_id, partial)


@triton.jit
def sumsq_rows_kernel(X_ptr, Sumsq_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row sum of squares across H. Same tiling and atomic add pattern as sum_rows_kernel.
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
    partial = tl.sum(x * x, axis=0)
    tl.atomic_add(Sumsq_ptr + row_id, partial)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, H):
    """
    Compute per-row mean and std (population, unbiased=False). Inputs: Sum_rows[S], Sumsq_rows[S].
    Outputs: Mean_rows[S], Std_rows[S].
    """
    row_id = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row_id)
    sumsq_val = tl.load(Sumsq_ptr + row_id)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean  # population variance
    # std is sqrt(var), Triton handles this
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    """
    Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation,
    write to z_ptr[0].
    """
    p = tl.load(p_ptr)
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

    # Constants for approximation
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

    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    tl.store(z_ptr, z_val)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, L, H, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: Out[row, col] = max(0, X[row, col] - Thresholds[row]).
    Grid is (S, tiles), each program handles one row and one tile.
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
    thresh = tl.load(Thresholds_ptr + row_id)
    y = x - thresh
    y = tl.maximum(y, 0.0)
    tl.store(Out_ptr + row_id * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Returns bfloat16 tensor. Default target_sparsity=0.1.
    """
    assert inputs.is_cuda, "ModelNew requires a CUDA tensor"
    inputs = inputs.contiguous()
    B, L, H = inputs.shape
    S = B * L

    # Work in float32 for stability
    X = inputs.to(torch.float32).view(S, H)

    # Allocate outputs for reductions
    Sum_rows = torch.zeros(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.zeros(S, dtype=torch.float32, device=inputs.device)
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Thresholds = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=inputs.device)

    # Constants
    BLOCK_SIZE = 1024
    grid_tiles = triton.cdiv(H, BLOCK_SIZE)

    # Launch sum and sumsq reductions
    sum_rows_kernel[(S, grid_tiles)](X, Sum_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    sumsq_rows_kernel[(S, grid_tiles)](X, Sumsq_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row
    mean_std_kernel[(S,)](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, H)

    # Compute z_scalar = _ndtri(target_sparsity) on device
    p = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p[0] = float(target_sparsity)
    z = torch.empty(1, dtype=torch.float32, device=inputs.device)
    ndtri_scalar_kernel[(1,)](p, z)

    # Thresholds per row
    Thresholds = Mean_rows + Std_rows * z

    # Gating
    gate_relu_kernel[(S, grid_tiles)](X, Thresholds, Out_flat, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16
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
