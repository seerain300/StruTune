import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # guard: only launch S programs
    # Compute base offset for this row
    base = row_id * H
    acc = 0.0
    for offs in range(0, H, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        ptrs = base + idx
        vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
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
        ptrs = base + idx
        vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + row_id, acc)


@triton.jit
def mean_std_kernel(Sum_ptr, Sumsq_ptr, Mean_ptr, Std_ptr, S, H):
    # Compute mean and std per row: unbiased=False (population std)
    for i in range(S):
        s = tl.load(Sum_ptr + i)
        ss = tl.load(Sumsq_ptr + i)
        mean = s / H
        var = ss / H - mean * mean
        std = tl.sqrt(var)
        tl.store(Mean_ptr + i, mean)
        tl.store(Std_ptr + i, std)


@triton.jit
def compute_thresholds_kernel(Mean_ptr, Std_ptr, z_scalar, Thresholds_ptr, S):
    # z_scalar is a 1-element device tensor; load it
    z = tl.load(z_scalar)
    for i in range(S):
        mean = tl.load(Mean_ptr + i)
        std = tl.load(Std_ptr + i)
        thr = mean + std * z
        tl.store(Thresholds_ptr + i, thr)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, L, H, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    base = row_id * H
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x_ptrs = base + offs
    thr = tl.load(Thresholds_ptr + row_id)
    x = tl.load(X_ptr + x_ptrs, mask=mask, other=0.0)
    y = tl.maximum(x - thr, 0.0)
    tl.store(Out_ptr + x_ptrs, y, mask=mask)


@triton.jit
def ndtri_vector_kernel(P_ptr, Z_ptr, size: tl.constexpr):
    # Compute inverse standard normal CDF for p[0] using Abramowitz-Stegun 7.1.26 approximation.
    # This is a 1-element vectorized kernel for robustness in Triton.
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

    p = tl.load(P_ptr)  # 1-element vector
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Compute z = sqrt(2) * erfinv(2p - 1) or use piecewise approximation
    # Implement piecewise approximation to match Abramowitz & Stegun 7.1.26
    # Lower region
    # For p < p_low: use rational approximation
    # Upper region
    # For p > p_high: use upper rational approximation
    # Central region: use polynomial in q = p - 0.5
    # We'll do masks and compute z accordingly.

    # Lower region
    # z_lower = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # Central region
    # z_mid = (((((a1*q^2 + a2)*q^2 + a3)*q^2 + a4)*q^2 + a5)*q^2 + a6)*q / (((((b1*q^2 + b2)*q^2 + b3)*q^2 + b4)*q^2 + b5)*q^2 + 1.0)
    # Upper region
    # z_upper = -z_lower (since it's symmetric)
    # Then combine masks and select

    # Compute q and z for each region
    # We'll use boolean masks and compute for all regions, then tl.where

    q_lower = tl.sqrt(-2.0 * tl.log(p))  # not used in lower path; kept for illustration
    # But actually in lower region we should use a different formula:
    # Let's recompute with correct logic. For Triton, we need to branch.

    # Triton supports tl.where but branching is vectorized; we can compute both approximations and select by masks.

    # Mask low
    mask_low = p < p_low
    # Mask high
    mask_high = p > p_high
    mask_mid = (~mask_low) & (~mask_high)

    # Compute rational approximations
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    q_mid = p - 0.5
    q_mid2 = q_mid * q_mid
    poly = (((((a1 * q_mid2 + a2) * q_mid2 + a3) * q_mid2 + a4) * q_mid2 + a5) * q_mid2 + a6) * q_mid
    denom = (((((b1 * q_mid2 + b2) * q_mid2 + b3) * q_mid2 + b4) * q_mid2 + b5) * q_mid2 + 1.0)
    z_mid = poly / denom

    # Select per mask
    z = tl.where(mask_low, z_low, tl.where(mask_high, z_up, z_mid))

    tl.store(Z_ptr, z)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    X = inputs.contiguous()
    B, L, H = X.shape
    S = B * L

    # Flatten to [S, H]
    X_flat = X.view(S, H)

    # Allocate per-row sums
    Sum = torch.empty(S, dtype=torch.float32, device=X.device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=X.device)

    # Launch row sum and sumsq
    BLOCK_SIZE = 1024
    grid_sum = (S,)
    row_sum_kernel[grid_sum](X_flat, Sum, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_sum](X_flat, Sumsq, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Mean and std
    Mean = torch.empty(S, dtype=torch.float32, device=X.device)
    Std = torch.empty(S, dtype=torch.float32, device=X.device)
    mean_std_kernel[(S,)](Sum, Sumsq, Mean, Std, S, H)

    # Compute z = _ndtri(target_sparsity) via Triton vector kernel (size=1)
    p_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    p_tensor[0] = float(target_sparsity)
    z_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    ndtri_vector_kernel[(1,)](p_tensor, z_tensor, size=1)

    # Thresholds per row: threshold = mean + std * z
    Thresholds = torch.empty(S, dtype=torch.float32, device=X.device)
    compute_thresholds_kernel[(S,)](Mean, Std, z_tensor, Thresholds, S)

    # Elementwise gating
    Out_flat = torch.empty_like(X_flat, dtype=torch.float32, device=X.device)
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X_flat, Thresholds, Out_flat, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    Out = Out_flat.view(B, L, H).to(torch.bfloat16)
    return Out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable