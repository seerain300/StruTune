import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * H
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        vals = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(Sum_ptr + row_id, acc)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * H
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        vals = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + row_id, acc)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    for i in range(0, S):
        sum_i = tl.load(Sum_ptr + i)
        sumsq_i = tl.load(Sumsq_ptr + i)
        mean_i = sum_i / H
        var_i = sumsq_i / H - mean_i * mean_i
        # guard against tiny negative due to rounding
        var_i = tl.maximum(var_i, 0.0)
        std_i = tl.sqrt(var_i)
        tl.store(Mean_ptr + i, mean_i)
        tl.store(Std_ptr + i, std_i)


@triton.jit
def ndtri_vector_kernel(P_ptr, Z_ptr, size: tl.constexpr):
    # Compute z = _ndtri(p) for p in (0, 1) using Abramowitz-Stegun 7.1.26 approximation
    # p is size=1 vector; Z_ptr stores scalar result
    # Load p
    p = tl.load(P_ptr)  # scalar-like
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    z = 0.0
    # Lower region
    if p < p_low:
        # q = sqrt(-2*log(p))
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
           ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    else:
        # Central region
        q = p - 0.5
        r = q * q
        z = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)

    tl.store(Z_ptr, z)


@triton.jit
def compute_thresholds_kernel(Mean_ptr, Std_ptr, Z_ptr, Thresholds_ptr, S):
    for i in range(0, S):
        mean_i = tl.load(Mean_ptr + i)
        std_i = tl.load(Std_ptr + i)
        z = tl.load(Z_ptr)  # scalar
        thr = mean_i + std_i * z
        tl.store(Thresholds_ptr + i, thr)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, L, H, BLOCK_SIZE: tl.constexpr):
    # 2D grid: row over S, feature tile over H
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    base = row_id * H
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
    thr = tl.load(Thresholds_ptr + row_id)
    y = tl.maximum(x - thr, 0.0)
    tl.store(Out_ptr + base + offs, y, mask=mask)


def _run_triton(X: torch.Tensor, target_sparsity: float, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    # X: [B, L, H] on CUDA
    B, L, H = X.shape
    S = B * L

    # Flatten to [S, H], cast to float32 for computation
    X_flat = X.contiguous().view(S, H).to(torch.float32)

    # 1) Per-row sum
    Sum = torch.empty(S, dtype=torch.float32, device=X.device)
    row_sum_kernel[(S,)](X_flat, Sum, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Per-row sum of squares
    Sumsq = torch.empty(S, dtype=torch.float32, device=X.device)
    row_sumsq_kernel[(S,)](X_flat, Sumsq, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 3) Compute mean and std per row
    Mean = torch.empty(S, dtype=torch.float32, device=X.device)
    Std = torch.empty(S, dtype=torch.float32, device=X.device)
    mean_std_kernel[(S,)](Sum, Sumsq, Mean, Std, S, H)

    # 4) Compute z = _ndtri(target_sparsity) using Triton vector kernel (size=1)
    p_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    p_tensor[0] = float(target_sparsity)
    z_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    ndtri_vector_kernel[(1,)](p_tensor, z_tensor, size=1)

    # 5) Compute per-row thresholds
    Thresholds = torch.empty(S, dtype=torch.float32, device=X.device)
    compute_thresholds_kernel[(S,)](Mean, Std, z_tensor, Thresholds, S)

    # 6) Elementwise gating
    Out_flat = torch.empty((S, H), dtype=torch.float32, device=X.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X_flat, Thresholds, Out_flat, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 7) Reshape and cast to bfloat16 to match original behavior
    Out = Out_flat.view(B, L, H).to(torch.bfloat16)
    return Out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable


def run(*args):
    return ModelNew()(*args)
