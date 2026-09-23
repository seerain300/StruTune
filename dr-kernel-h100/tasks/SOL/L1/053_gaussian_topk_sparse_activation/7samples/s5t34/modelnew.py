import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_row_sum(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row; iterate over H in tiles and accumulate sum
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Accumulator
    acc = 0.0
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        # Reduce tile into scalar
        # Note: tl.sum returns a scalar
        acc += tl.sum(x, axis=0)
        col += BLOCK_SIZE
    # Store per-row sum
    tl.store(Sum_ptr + row, acc)


@triton.jit
def _reduce_row_sumsq(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row; iterate over H in tiles and accumulate sum of squares
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE
    tl.store(Sumsq_ptr + row, acc)


@triton.jit
def _compute_mean_std(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # Vector kernel: per-row mean and std
    row = tl.program_id(0)
    sum_row = tl.load(Sum_ptr + row)
    sumsq_row = tl.load(Sumsq_ptr + row)
    mean = sum_row / H
    var = sumsq_row / H - mean * mean
    # Clamp variance to non-negative to avoid small negative due to floating-point
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def _ndtri_scalar_kernel(p, z_ptr):
    # Compute inverse standard normal CDF for p (Abramowitz-Stegun 7.1.26 approximation)
    # Writes result to z_ptr[0].
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

    # Piecewise approximation
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Store result
    tl.store(z_ptr, z)


@triton.jit
def _gate_row_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row; iterate across H in tiles and apply gating: y = max(0, x - threshold)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    threshold = tl.load(Thresholds_ptr + row)
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Out_ptr + row * H + idx, y, mask=mask)
        col += BLOCK_SIZE


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std, then applies y = max(0, x - (mean + std * _ndtri(target_sparsity)))
    and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Flatten to [S, H], compute in float32
    B, L, H = inputs.shape
    S = B * L
    X = inputs.to(torch.float32).view(S, H)

    # Allocate per-row sums
    Sum_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # Launch reduction kernels with correct 2D grid
    BLOCK_SIZE = 1024
    grid_red = (S, triton.cdiv(H, BLOCK_SIZE))
    _reduce_row_sum[grid_red](X, Sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _reduce_row_sumsq[grid_red](X, Sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row
    Mean_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=inputs.device)
    _compute_mean_std[(S,)](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, S, H)

    # Compute z = _ndtri(target_sparsity) on device via Triton scalar kernel
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_scalar_kernel(target_sparsity, z_tensor)  # single program writes z

    # Compute per-row thresholds
    thresholds = Mean_rows + Std_rows * z_tensor  # broadcast 1-element tensor

    # Allocate output and run elementwise gating
    Out = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
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