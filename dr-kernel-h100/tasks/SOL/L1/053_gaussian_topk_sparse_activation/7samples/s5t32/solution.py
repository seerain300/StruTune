import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_scalar_kernel(p, z_ptr):
    # Compute inverse standard normal CDF for p (float32) using Abramowitz-Stegun 7.1.26 approximation.
    # This kernel writes the result to z_ptr[0] as a 1-element device tensor.
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

    # Piecewise selection
    mask_low = p < p_low
    mask_high = p > p_high
    # Triton's tl.where works with scalars; build the result
    # Start with z_low for mask_low, z_mid for central, z_high for upper
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(mask_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    # Store result
    tl.store(z_ptr, z_val)


@triton.jit
def _reduce_row_sum(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row. Accumulate sum across H in tiles of BLOCK_SIZE.
    row_id = tl.program_id(0)
    # Guard: if row_id >= S, return
    if row_id >= S:
        return
    # Compute base pointer for this row
    # X_ptr is [S, H] flattened; row start = row_id * H
    base = row_id * H
    total = 0.0
    # Iterate tiles across H
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        ptrs = X_ptr + base + offs
        x = tl.load(ptrs, mask=mask, other=0.0)
        # accumulate in fp32
        x = x.to(tl.float32)
        total += tl.sum(x, axis=0)
    tl.store(Sum_ptr + row_id, total)


@triton.jit
def _reduce_row_sumsq(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= S:
        return
    base = row_id * H
    total = 0.0
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        ptrs = X_ptr + base + offs
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.store(Sumsq_ptr + row_id, total)


@triton.jit
def _compute_mean_std(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # Vector kernel: one program per row, compute mean and std.
    row_id = tl.program_id(0)
    if row_id >= S:
        return
    sum_row = tl.load(Sum_ptr + row_id)
    sumsq_row = tl.load(Sumsq_ptr + row_id)
    # mean
    mean = sum_row / H
    # var = E[x^2] - (E[x])^2 (population variance)
    var = sumsq_row / H - mean * mean
    # guard var from tiny negatives due to fp roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def _gate_row_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row, gate across H in tiles.
    row_id = tl.program_id(0)
    if row_id >= S:
        return
    base_x = row_id * H
    threshold = tl.load(Thresholds_ptr + row_id)
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + base_x + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Out_ptr + base_x + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()
    B, L, H = inputs.shape
    S = B * L

    # Cast to float32 and flatten to [S, H] for Triton kernels
    X = inputs.to(torch.float32)
    X_flat = X.view(S * H)  # flattened 1D to simplify pointer math in kernels (we'll recompute row bases)
    # Instead, keep as 2D view: (S, H)
    X2d = X.view(S, H)

    # Allocate reduction outputs
    Sum_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # Launch reductions: 2D grid over rows and tiles of H
    BLOCK_SIZE = 1024
    grid = (S, triton.cdiv(H, BLOCK_SIZE))
    _reduce_row_sum[grid](X2d, Sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _reduce_row_sumsq[grid](X2d, Sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    _compute_mean_std[(S,)](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, S, H)

    # Compute z = _ndtri(target_sparsity) on device (1-element tensor)
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_scalar_kernel(float(target_sparsity), z_tensor)  # pass scalar directly

    # Compute per-row thresholds
    thresholds = Mean_rows + Std_rows * z_tensor  # broadcast 1-element tensor

    # Allocate output and run elementwise gating
    Out2d = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    _gate_row_kernel[(S,)](X2d, thresholds, Out2d, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out2d.view(B, L, H).to(torch.bfloat16)
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
