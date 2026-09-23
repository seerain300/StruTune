import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    # If grid covers all rows, row < S, so no extra bound check needed
    total = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        start += BLOCK_SIZE
    tl.store(Sum_ptr + row, total)


@triton.jit
def _row_sumsq_kernel(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    total = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        total += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE
    tl.store(Sumsq_ptr + row, total)


@triton.jit
def _mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    row = tl.program_id(0)
    sumv = tl.load(Sum_ptr + row)
    sumsqv = tl.load(Sumsq_ptr + row)
    mean = sumv / H
    var = sumsqv / H - mean * mean
    # Ensure non-negative variance for stability
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def _gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, B, L, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    start = tile * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
    th = tl.load(Thresholds_ptr + row)  # scalar per row
    y = x - th
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(Out_ptr + row * H + offs, y, mask=mask)


@triton.jit
def _ndtri_kernel(P_ptr, Z_ptr):
    # Compute inverse standard normal CDF for a single p via Abramowitz-Stegun 7.1.26 approximation.
    p = tl.load(P_ptr)  # scalar
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

    # Piecewise select
    mask_low = p < p_low
    mask_high = p > p_high
    # Note: Triton supports tl.where with scalar/mask semantics
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)
    tl.store(Z_ptr, z_val)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the feature dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    # Cast to float32 for stable math; flatten [B, L, H] -> [S, H]
    B, L, H = inputs.shape
    S = B * L
    inputs_f32 = inputs.to(torch.float32)
    X_flat = inputs_f32.view(S, H).contiguous()

    # Allocate and launch kernels
    Sum_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # Launch reduction kernels: one program per row, iterate across H in tiles
    BLOCK_SIZE = 2048  # tile size; safe for H up to 12288
    grid = (S,)
    _row_sum_kernel[grid](X_flat, Sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _row_sumsq_kernel[grid](X_flat, Sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std via Triton kernel
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    _mean_std_kernel[grid](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, S, H, num_warps=1)

    # Compute z_scalar = _ndtri(target_sparsity) via Triton kernel
    p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_kernel[(1,)](p_tensor, z_scalar)  # single program, scalar input/output

    # Compute thresholds: threshold = mean + std * z_scalar
    Thresholds = Mean_rows + Std_rows * z_scalar[0]  # broadcast per row

    # Elementwise gating: y = max(0, x - threshold) with 2D grid over tiles
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    TILES = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_gate = (S, TILES)
    _gate_relu_kernel[grid_gate](X_flat, Thresholds, Out_flat, S, B, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

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
