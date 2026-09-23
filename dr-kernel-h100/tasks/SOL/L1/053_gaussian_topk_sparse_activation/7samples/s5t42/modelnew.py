import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_atomic_kernel(X_ptr, Sum_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row sum across H using a 2D grid: (row id, tile id).
    Each program processes BLOCK_SIZE elements of the row, masks tail, and atomic_adds to Sum_ptr[row].
    X_ptr: [S, H], contiguous, S = B * L.
    Sum_ptr: [S], float32.
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    # Each program processes a contiguous chunk of BLOCK_SIZE features
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    # Compute base pointer for this row
    row_base = row_id * H
    x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)  # sum across this tile
    tl.atomic_add(Sum_ptr + row_id, partial)


@triton.jit
def _sumsq_rows_atomic_kernel(X_ptr, Sumsq_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row sum of squares across H using a 2D grid: (row id, tile id).
    Each program processes BLOCK_SIZE elements of the row, masks tail, and atomic_adds x^2 to Sumsq_ptr[row].
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    row_base = row_id * H
    x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
    x2 = x * x
    partial = tl.sum(x2, axis=0)
    tl.atomic_add(Sumsq_ptr + row_id, partial)


@triton.jit
def _compute_mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise per-row computation:
      mean = Sum / H
      var  = Sumsq / H - mean^2   (population variance, unbiased=False)
      std  = sqrt(var)
    """
    row_id = tl.program_id(0)
    s = tl.load(Sum_ptr + row_id)
    ss = tl.load(Sumsq_ptr + row_id)
    mean = s / H
    var = ss / H - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def _ndtri_scalar_kernel(p_ptr, z_ptr):
    """
    Compute inverse standard normal CDF for a single probability p_ptr[0].
    Uses Abramowitz-Stegun 7.1.26 approximation.
    Writes result to z_ptr[0] as float32.
    """
    p = tl.load(p_ptr)
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

    # Coefficients for lower/central/upper region approximations
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
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
def _gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: Out[row, j] = max(0, X[row, j] - Thresholds[row]).
    2D grid over rows and tiles of BLOCK_SIZE features per program.
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    row_base = row_id * H
    x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
    threshold = tl.load(Thresholds_ptr + row_id)
    y = tl.maximum(x - threshold, 0.0)
    tl.store(Out_ptr + row_base + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dim, then applies
    y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure CUDA and contiguous; default to float32 for numerics
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()
    B, L, H = inputs.shape
    S = B * L

    X = inputs.to(torch.float32).view(S, H)  # [S, H]

    # Allocate reduction buffers
    Sum_rows = torch.zeros(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.zeros(S, dtype=torch.float32, device=inputs.device)
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # 1) Compute per-row sum and sumsq via atomic adds over tiles
    BLOCK_SIZE = 1024
    grid_tiles = triton.cdiv(H, BLOCK_SIZE)
    _sum_rows_atomic_kernel[(S, grid_tiles)](X, Sum_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _sumsq_rows_atomic_kernel[(S, grid_tiles)](X, Sumsq_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Compute mean and std per row
    _compute_mean_std_kernel[(S,)](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)

    # 3) Compute per-row thresholds: mean + std * _ndtri(target_sparsity)
    sp_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device).view(1)
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_scalar_kernel[(1,)](sp_tensor, z_tensor)
    Thresholds = Mean_rows + Std_rows * z_tensor  # shape [S], broadcast along H

    # 4) Apply gating: y = max(0, x - threshold), 2D grid over rows and tiles
    Out = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    _gate_relu_kernel[(S, triton.cdiv(H, BLOCK_SIZE))](X, Thresholds, Out, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    return Out.view(B, L, H).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable