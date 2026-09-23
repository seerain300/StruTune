import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(X, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row; accumulate sum over H in tiles
    row_id = tl.program_id(0)
    # Guard against out-of-range row_id (in case grid > S), though we set grid=S
    if row_id >= S:
        return
    # Accumulate sum for this row
    acc = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row_id * H + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
        start += BLOCK_SIZE
    tl.store(Sum_ptr + row_id, acc)


@triton.jit
def _row_sumsq_kernel(X, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= S:
        return
    acc = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row_id * H + offs, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE
    tl.store(Sumsq_ptr + row_id, acc)


@triton.jit
def _compute_mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # Compute mean and std per row
    for i in range(S):
        s = tl.load(Sum_ptr + i)
        ss = tl.load(Sumsq_ptr + i)
        mean = s / H
        var = ss / H - mean * mean
        # Ensure non-negative due to numerical noise
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(Mean_ptr + i, mean)
        tl.store(Std_ptr + i, std)


@triton.jit
def _compute_thresholds_kernel(Mean_ptr, Std_ptr, z_scalar, Thresholds_ptr, S):
    # threshold = mean + std * z_scalar
    for i in range(S):
        mean = tl.load(Mean_ptr + i)
        std = tl.load(Std_ptr + i)
        th = mean + std * z_scalar
        tl.store(Thresholds_ptr + i, th)


@triton.jit
def _gate_relu_kernel(X, Thresholds_ptr, Out_ptr, S, L, H, z_scalar, BLOCK_SIZE: tl.constexpr):
    # 2D grid: (row_id, tile_id) across H
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    if row_id >= S:
        return
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load x for this row
    x = tl.load(X + row_id * H + offs, mask=mask, other=0.0)

    # Load threshold for this row
    # Thresholds_ptr has length S, one per row
    th = tl.load(Thresholds_ptr + row_id)

    # Gate: y = max(0, x - th)
    y = x - th
    y = tl.maximum(y, 0.0)

    tl.store(Out_ptr + row_id * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation:
    - Computes per-row mean and std (population), threshold = mean + std * _ndtri(target_sparsity),
      then applies y = max(0, x - threshold), returns bfloat16.
    """
    # Ensure CUDA tensor and contiguous layout
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    # Flatten to [S, H]
    B, L, H = inputs.shape
    S = B * L
    X = inputs.view(S, H).to(torch.float32)

    # Output buffer for gating
    Out = torch.empty_like(X, dtype=torch.float32, device=inputs.device)

    # Allocate intermediates
    Sum = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Mean = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Thresholds = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # Launch row reductions
    BLOCK_SIZE = 1024
    grid = (S,)
    _row_sum_kernel[grid](X, Sum, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _row_sumsq_kernel[grid](X, Sumsq, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row
    _compute_mean_std_kernel[(S,)](Sum, Sumsq, Mean, Std, S, H)

    # Compute z_scalar as inverse normal CDF of target_sparsity using A&S 7.1.26
    # For target_sparsity=0.1, z_scalar ≈ 1.2815515655446014
    # Implement a simple Triton kernel for robustness; use a 1-element vector kernel.
    p = torch.empty(1, dtype=torch.float32, device=inputs.device)
    p[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs.device)
    # A&S 7.1.26 approximation; we run one program and write to z_scalar
    # Constants
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

    mask_low = p < p_low
    mask_high = p > p_high
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(p > p_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)
    z_scalar[0] = z_val  # store 1-element result

    # Compute per-row thresholds: threshold = mean + std * z_scalar
    _compute_thresholds_kernel[(S,)](Mean, Std, z_scalar[0], Thresholds, S)

    # Elementwise gating
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    _gate_relu_kernel[grid_gate](X, Thresholds, Out, S, B, H, z_scalar[0], BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        # Ensure CUDA tensor for Triton
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable


def run(*args):
    return ModelNew()(*args)
