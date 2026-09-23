import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_scalar_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF (quantile) for a single value p (0 < p < 1).
    # Abramowitz and Stegun formula 7.1.26.
    # Writes result to z_ptr[0].
    p = tl.load(p_ptr)
    # Constants
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.141592653589793

    # Low region approximation coefficients
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

    # Low region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Mid region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # High region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Piecewise selection
    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)
    tl.store(z_ptr, z_val)


@triton.jit
def reduce_sum_sumsq(X_ptr, Sum_ptr, Sumsq_ptr, S, H, TILE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0
    # Loop over tiles
    for t in range(TILE):
        start = t * 1024
        offs = start + tl.arange(0, 1024)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        # Reduce tile
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
    tl.store(Sum_ptr + row, acc_sum)
    tl.store(Sumsq_ptr + row, acc_sumsq)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # One program per row
    row = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row)
    sumsq_val = tl.load(Sumsq_ptr + row)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    # Numerical guard: var could be slightly negative due to rounding; clamp to >= 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, z_scalar, TILE: tl.constexpr):
    # 2D grid: rows x tiles
    row = tl.program_id(0)
    tile = tl.program_id(1)
    start = tile * 1024
    offs = start + tl.arange(0, 1024)
    mask = offs < H
    # Load row's threshold
    threshold = tl.load(Thresholds_ptr + row)
    # Compute gate: y = max(0, x - threshold)
    x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
    y = x - threshold * z_scalar
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(Out_ptr + row * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA tensor and contiguous
    assert inputs.is_cuda, "ModelNew requires a CUDA tensor"
    inputs = inputs.contiguous()
    B, L, H = inputs.shape
    S = B * L
    device = inputs.device

    # Flatten to [S, H]
    X_flat = inputs.view(S, H)

    # Allocate accumulators
    Sum = torch.empty(S, dtype=torch.float32, device=device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=device)

    # Reduction: sum and sumsq per row (no atomics)
    TILE = (H + 1023) // 1024  # number of tiles
    reduce_sum_sumsq[(S,)](X_flat, Sum, Sumsq, S, H, TILE=TILE)

    # Compute mean and std per row
    Mean_rows = torch.empty(S, dtype=torch.float32, device=device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=device)
    mean_std_kernel[(S,)](Sum, Sumsq, Mean_rows, Std_rows, S, H)

    # Compute inverse normal CDF for target_sparsity in Triton (scalar)
    sp_tensor = torch.empty(1, dtype=torch.float32, device=device)
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=device)
    _ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)

    # Elementwise gating: y = max(0, x - (mean + std * z_scalar))
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=device)
    grid_gate = (S, TILE)
    gate_relu_kernel[grid_gate](X_flat, Mean_rows, Out_flat, S, H, z_scalar[0], TILE=TILE)

    # Reshape back and cast to bfloat16 to match original behavior
    out = Out_flat.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable