import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_row_sum(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    # Accumulator for this row
    acc = 0.0
    # Iterate over H in tiles
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row base pointer: X is [S, H] with row-major contiguous layout
        row_base = X_ptr + row * H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        # Sum masked values and accumulate
        acc += tl.sum(x, axis=0)
    # Store sum for this row
    tl.store(Sum_ptr + row, acc)


@triton.jit
def _reduce_row_sumsq(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        row_base = X_ptr + row * H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    tl.store(Sumsq_ptr + row, acc)


@triton.jit
def _compute_mean_std(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # Vector kernel: one program per row
    row = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row)
    sumsq_val = tl.load(Sumsq_ptr + row)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to FP rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def _ndtri_scalar_kernel(p, z_ptr):
    # Compute inverse standard normal CDF for p using Abramowitz-Stegun 7.1.26 approximation.
    # p: scalar float, z_ptr: 1-element tensor to write result.
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

    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

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
    # Use tl.where for proper scalar selection
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    tl.store(z_ptr, z_val)


@triton.jit
def _gate_row_kernel(X_ptr, Thresholds_ptr, Y_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row, iterate tiles across H
    row = tl.program_id(0)
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        row_base = X_ptr + row * H
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        thr = tl.load(Thresholds_ptr + row)  # scalar threshold for this row
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + row * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Cast to float32 for stable math; flatten [B, L, H] -> [S, H]
    B, L, H = inputs.shape
    S = B * L
    inputs_f32 = inputs.to(torch.float32)
    X = inputs_f32.view(S, H)

    # 1) Reduce: per-row sum and sum of squares
    Sum_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    BLOCK_SIZE = 1024
    grid = (S,)
    _reduce_row_sum[grid](X, Sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _reduce_row_sumsq[grid](X, Sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Compute mean and std per row
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    _compute_mean_std[grid](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, S, H)

    # 3) Compute z = _ndtri(target_sparsity) on device
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    # Pass target_sparsity as a scalar argument; Triton will treat it as a Python float
    _ndtri_scalar_kernel(target_sparsity, z_tensor)  # single program handles scalar

    # 4) Compute per-row thresholds
    thresholds = Mean_rows + Std_rows * z_tensor  # broadcasts the 1-element tensor

    # 5) Apply gating y = max(0, x - threshold)
    Out = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    grid_gate = (S,)
    _gate_row_kernel[grid_gate](X, thresholds, Out, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable