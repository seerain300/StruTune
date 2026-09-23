import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Initialize running sum
    total = 0.0
    col = 0
    # Iterate over tiles
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        # Reduce this tile to a scalar
        tile_sum = tl.sum(tl.where(mask, x, 0.0))
        total += tile_sum
        col += BLOCK_SIZE
    tl.store(Sum_ptr + row, total)


@triton.jit
def sumsq_kernel(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    total = 0.0
    col = 0
    while col < H:
        idx = col + offs
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        tile_sum = tl.sum(tl.where(mask, x * x, 0.0))
        total += tile_sum
        col += BLOCK_SIZE
    tl.store(Sumsq_ptr + row, total)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # One program per row
    row = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row)
    sumsq_val = tl.load(Sumsq_ptr + row)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    # Guard for negative due to numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def ndtri_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation.
    # p_ptr: 1-element tensor (float32), z_ptr: 1-element tensor (float32)
    p = tl.load(p_ptr)
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.1415927

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
def compute_thresholds_kernel(Mean_ptr, Std_ptr, z_scalar, Thresholds_ptr, S):
    row = tl.program_id(0)
    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    thresh = mean + std * z_scalar
    tl.store(Thresholds_ptr + row, thresh)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, z_scalar, BLOCK_SIZE: tl.constexpr):
    # 2D grid: (rows, tiles)
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
    thresh = tl.load(Thresholds_ptr + row)
    y = x - thresh - z_scalar
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(Out_ptr + row * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure CUDA tensor for Triton; preserve original device afterward
    orig_device = inputs.device
    inputs_cuda = inputs if inputs.is_cuda else inputs.cuda()
    inputs_cuda = inputs_cuda.contiguous()
    B, L, H = inputs_cuda.shape
    S = B * L

    # Cast to float32 and flatten to [S, H]
    X = inputs_cuda.to(torch.float32)
    X = X.view(S, H)

    # 1) Compute sum and sum of squares per row
    Sum = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    BLOCK_SIZE = 1024
    sum_kernel[(S,)](X, Sum, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    sumsq_kernel[(S,)](X, Sumsq, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Compute mean and std per row
    Mean = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Std = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    mean_std_kernel[(S,)](Sum, Sumsq, Mean, Std, S, H)

    # 3) Compute inverse normal CDF z_scalar for target_sparsity=0.1 via Triton
    p = torch.empty(1, dtype=torch.float32, device=inputs_cuda.device)
    p[0] = float(target_sparsity)  # default 0.1
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs_cuda.device)
    ndtri_kernel[(1,)](p, z_scalar)

    # 4) Compute per-row thresholds
    Thresholds = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    compute_thresholds_kernel[(S,)](Mean, Std, z_scalar[0], Thresholds, S)

    # 5) Elementwise gating y = max(0, x - threshold)
    Out = torch.empty((S, H), dtype=torch.float32, device=inputs_cuda.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, Thresholds, Out, S=S, H=H, z_scalar=float(target_sparsity), BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out.view(B, L, H).to(torch.bfloat16)

    # Move back to original device if needed
    if orig_device.type != 'cuda':
        out = out.cpu()
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable