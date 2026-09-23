import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_vector_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for p_ptr[0] using Abramowitz-Stegun 7.1.26 approximation.
    # p_ptr: 1-element tensor (float32), z_ptr: 1-element tensor (float32)
    p = tl.load(p_ptr)
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

    # Select piecewise
    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)

    tl.store(z_ptr, z_val)


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    # One program per (b, l) row: accumulate sum over H in tiles
    row_id = tl.program_id(0)
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(X_ptr + row_id * H + idx, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(Sum_ptr + row_id, acc)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(X_ptr + row_id * H + idx, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    tl.store(Sumsq_ptr + row_id, acc)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, H: tl.int32):
    row_id = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row_id)
    sumsq_val = tl.load(Sumsq_ptr + row_id)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def gate_relu_kernel(X_ptr, Mean_ptr, Std_ptr, z_scalar, Out_ptr, B: tl.int32, L: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    # 2D grid over rows (B*L) and tiles across H
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Compute threshold for this row
    mean = tl.load(Mean_ptr + row_id)
    std = tl.load(Std_ptr + row_id)
    threshold = mean + std * z_scalar

    x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
    y = tl.maximum(x - threshold, 0.0)  # ReLU
    tl.store(Out_ptr + row_id * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    B, L, H = inputs.shape
    # Cast to float32 for stable math; flatten rows conceptually (row-major [B*L, H])
    X = inputs.to(torch.float32)

    # Output flat [B*L, H]
    Out = torch.empty((B * L, H), dtype=torch.float32, device=inputs.device)

    # 1) Compute z_scalar = _ndtri(target_sparsity) on device
    p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p_tensor[0] = float(target_sparsity)
    z_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
    _ndtri_vector_kernel[(1,)](p_tensor, z_tensor)  # single program handles scalar

    # 2) Row-wise sum and sumsq
    Sum = torch.empty(B * L, dtype=torch.float32, device=inputs.device)
    Sumsq = torch.empty(B * L, dtype=torch.float32, device=inputs.device)
    BLOCK_SIZE = 1024  # tile size for reduction; robust for H up to 12288
    grid_reduce = (B * L,)
    row_sum_kernel[grid_reduce](X, Sum, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, Sumsq, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 3) Compute mean and std per row
    Mean = torch.empty(B * L, dtype=torch.float32, device=inputs.device)
    Std = torch.empty(B * L, dtype=torch.float32, device=inputs.device)
    mean_std_kernel[grid_reduce](Sum, Sumsq, Mean, Std, H)

    # 4) Elementwise gating: y = max(0, x - (mean + std * z_scalar))
    grid_gate = (B * L, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, Mean, Std, z_tensor[0], Out, B, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

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