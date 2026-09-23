import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    sum_val = 0.0
    # Iterate over the feature dimension in chunks
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
        # x is float32; sum values
        sum_val += tl.sum(x, axis=0)
    tl.store(Sum_ptr + row_id, sum_val)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    sumsq_val = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row_id * H + offs, mask=mask, other=0.0)
        sumsq_val += tl.sum(x * x, axis=0)
    tl.store(Sumsq_ptr + row_id, sumsq_val)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, H: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row_id)
    sumsq_val = tl.load(Sumsq_ptr + row_id)
    h = H  # constexpr
    mean = sum_val / h
    var = sumsq_val / h - mean * mean
    # std = sqrt(var); ensure non-negative due to numerical rounding
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def ndtri_vector_kernel(p_ptr, z_ptr):
    # Compute inverse standard normal CDF for a single p via Abramowitz-Stegun 7.1.26 approximation.
    # p_ptr: 1-element tensor (float32), z_ptr: 1-element tensor (float32)
    p = tl.load(p_ptr)
    p_low = 2.425e-2
    p_high = 1.0 - p_low
    pi = 3.1415927

    # Constants for approximation
    a1 = -3.9696830e+01
    a2 = 2.2094609e+02
    a3 = -2.7592851e+02
    a4 = 1.3835775e+02
    a5 = -3.0664799e+01
    a6 = 2.5066283e+00

    b1 = -5.4476100e+01
    b2 = 1.6158584e+02
    b3 = -1.5569898e+02
    b4 = 6.6801312e+01
    b5 = -1.3280682e+01

    c1 = -7.7848940e-03
    c2 = -3.2239646e-01
    c3 = -2.4007583e+00
    c4 = -2.5497325e+00
    c5 = 4.3746641e+00
    c6 = 2.9381640e+00

    d1 = 7.7846957e-03
    d2 = 3.2246713e-01
    d3 = 2.4451341e+00
    d4 = 3.7544087e+00

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
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr,
                     S: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
                     z_scalar: tl.float32, BLOCK_SIZE: tl.constexpr):
    # 2D grid: program_id(0) = row index, program_id(1) = tile index along H
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    col_start = tile_id * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load this row's threshold (scalar)
    threshold = tl.load(Thresholds_ptr + row_id)  # already computed in host code

    # Compute pointer for this row
    base = X_ptr + row_id * H

    # Load x tile
    x = tl.load(base + offs, mask=mask, other=0.0)

    # Compute gated output: y = max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store result
    tl.store(Out_ptr + row_id * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dim, then applies
    y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Flatten to [S, H]
    B, L, H = inputs.shape
    S = B * L
    X = inputs.to(torch.float32).view(S, H)

    # Allocate intermediates
    Sum_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    Sumsq_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    Mean_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    Std_rows = torch.empty(S, dtype=torch.float32, device=X.device)

    # 1) Compute row sums
    BLOCK_SIZE = 1024
    grid_sum = (S,)
    row_sum_kernel[grid_sum](X, Sum_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Compute row sums of squares
    grid_sumsq = (S,)
    row_sumsq_kernel[grid_sumsq](X, Sumsq_rows, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 3) Compute per-row mean and std (population, unbiased=False)
    grid_ms = (S,)
    mean_std_kernel[grid_ms](Sum_rows, Sumsq_rows, Mean_rows, Std_rows, H=H, num_warps=1)

    # 4) Compute z_scalar = _ndtri(target_sparsity) on device (single value)
    p_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    p_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=X.device)
    grid_ndtri = (1,)
    ndtri_vector_kernel[grid_ndtri](p_tensor, z_scalar)

    # Precompute thresholds per row: threshold = mean + std * z_scalar
    # Ensure z_scalar is a scalar tensor on device
    z_scalar_t = z_scalar[0]  # scalar value
    Thresholds = Mean_rows + Std_rows * z_scalar_t  # elementwise in PyTorch

    # 5) Elementwise gating: y = max(0, x - threshold)
    Out_flat = torch.empty_like(X, dtype=torch.float32, device=X.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](
        X, Thresholds, Out_flat,
        S=S, L=L, H=H,
        z_scalar=float(z_scalar_t), BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )

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
