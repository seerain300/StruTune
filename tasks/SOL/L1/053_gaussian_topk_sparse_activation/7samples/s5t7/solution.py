import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row sum across last dim (features)
@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
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
    pid = tl.program_id(axis=0)  # one program per row
    row_start = pid * H
    acc = 0.0
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        vals = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + pid, acc)


# Triton kernel: compute per-row std from sums and sumsq (all float32)
@triton.jit
def std_rows_kernel(Sum_ptr, Sumsq_ptr, Std_ptr, S: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    sum_row = tl.load(Sum_ptr + pid)
    sumsq_row = tl.load(Sumsq_ptr + pid)
    mean = sum_row / H
    var = sumsq_row / H - mean * mean
    std = tl.sqrt(var)  # population std (unbiased=False)
    tl.store(Std_ptr + pid, std)


# Triton scalar kernel: compute inverse standard normal CDF via Abramowitz-Stegun 7.1.26
# z = _ndtri(p) for p in (0, 1), returns negative for sparsity > 0.5.
@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    p = tl.load(p_ptr)  # scalar float32
    # Piecewise constants
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
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select result based on mask
    z = tl.zeros((), dtype=tl.float32)
    z = tl.where(mask_low, z_low, z)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store scalar
    tl.store(z_ptr, z)


# Triton kernel: compute per-row thresholds from std and mean and scalar z
@triton.jit
def compute_thresholds_kernel(std_ptr, mean_ptr, z_scalar_ptr, thresholds_ptr, S: tl.constexpr):
    # z_scalar_ptr is a 1-element tensor; load as scalar
    z = tl.load(z_scalar_ptr)
    for i in range(0, S):
        std_i = tl.load(std_ptr + i)
        mean_i = tl.load(mean_ptr + i)
        tl.store(thresholds_ptr + i, mean_i + std_i * z)


# Triton elementwise kernel: y = max(0, x - threshold), per-row thresholds broadcast along features
@triton.jit
def gate_relu_kernel(X_ptr, thresholds_ptr, Out_ptr, S: tl.constexpr, L: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # axis 0: row id (0..S-1), axis 1: tile id along features
    pid_row = tl.program_id(axis=0)
    pid_tile = tl.program_id(axis=1)
    row_start = pid_row * H
    off = pid_tile * BLOCK_SIZE
    idx = off + tl.arange(0, BLOCK_SIZE)
    mask = idx < H

    # load x
    x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
    # load threshold for this row
    threshold = tl.load(thresholds_ptr + pid_row)
    # gating
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(Out_ptr + row_start + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    # Flatten to [S, H]
    B, L, H = inputs.shape
    S = B * L
    X = inputs.contiguous().view(S, H).to(torch.float32)

    # Allocate per-row sums
    sum_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=X.device)

    # Launch reduction kernels: one program per row
    BLOCK_SIZE = 1024  # reasonable for H up to 16384 and beyond via loop
    grid_reduce = (S,)
    row_sum_kernel[grid_reduce](X, sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute per-row std (population) using Triton vector kernel
    std_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    std_rows_kernel[(S,)](sum_rows, sumsq_rows, std_rows, S, H)

    # Compute z = _ndtri(target_sparsity) on device via Triton scalar kernel
    sp_tensor = torch.empty(1, dtype=torch.float32, device=X.device)  # 1-element tensor on device
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=X.device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)  # single program handles scalar

    # Compute per-row thresholds on device using Triton
    thresholds = torch.empty(S, dtype=torch.float32, device=X.device)
    compute_thresholds_kernel[(S,)](std_rows, sum_rows / H, z_scalar, thresholds, S)

    # Allocate output
    out = torch.empty(S, H, dtype=torch.float32, device=X.device)

    # Launch elementwise gating kernel: 2D grid over rows and feature tiles
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, thresholds, out, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = out.view(B, L, H).to(torch.bfloat16)
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
