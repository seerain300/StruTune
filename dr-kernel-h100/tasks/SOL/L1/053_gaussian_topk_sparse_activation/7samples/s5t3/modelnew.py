import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row sum across last dim (features)
@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * H
    acc = 0.0
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        vals = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(Sum_ptr + pid, acc)


# Triton kernel: compute per-row sum of squares across last dim
@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * H
    acc = 0.0
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        vals = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + pid, acc)


# Triton scalar kernel: compute inverse standard normal CDF (Abramowitz-Stegun approximation)
@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    # Load p as a 1-element tensor
    p = tl.load(p_ptr)
    # Constants for approximation
    p_low = 0.02425
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

    # Masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store result into 1-element output tensor
    tl.store(z_ptr, z)


# Triton elementwise kernel: y = max(0, x - threshold), threshold per (b,l) row
@triton.jit
def gate_relu_kernel(X_ptr, Threshold_ptr, Out_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(axis=0)
    l = tl.program_id(axis=1)
    row_start = b * L * H + l * H
    threshold = tl.load(Threshold_ptr + b * L + l)
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - threshold
        # ReLU: max(0, y)
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(Out_ptr + row_start + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float = 0.1) -> torch.Tensor:
    # Ensure CUDA and contiguous; compute in float32 for stability
    device = inputs.device
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()
    B, L, H = inputs.shape
    S = B * L

    # Cast to float32 for Triton kernels
    X = inputs.to(torch.float32)

    # Allocate per-row sums
    sum_rows = torch.empty(S, dtype=torch.float32, device=device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=device)

    # Launch reduction kernels: one program per row
    BLOCK_SIZE = 1024  # tune as needed; works for H up to 16384 and beyond via loop
    grid_reduce = (S,)
    row_sum_kernel[grid_reduce](X, sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and variance in float32
    mean = sum_rows / H
    sumsq = sumsq_rows
    var = sumsq / H - mean * mean
    std = torch.sqrt(var)  # std >= 0 by construction

    # Compute z = _ndtri(target_sparsity) on device via Triton scalar kernel
    sp_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
    z_tensor = torch.empty((), dtype=torch.float32, device=device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_tensor)  # single program handles scalar

    # Prepare per-row thresholds: threshold[b,l] = mean[b,l] + std[b,l] * z
    # Thresholds are 1D length S (flattened [B*L]); compute in host, but this is minimal
    thresholds = mean + std * z_tensor  # broadcasts z_tensor (scalar) across rows

    # Allocate output
    out = torch.empty_like(inputs, dtype=torch.float32)

    # Launch elementwise gating kernel: 2D grid over (b,l)
    grid_gate = (B, L)
    gate_relu_kernel[grid_gate](X, thresholds, out, B, L, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Cast back to bfloat16 to match original function’s return type
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)