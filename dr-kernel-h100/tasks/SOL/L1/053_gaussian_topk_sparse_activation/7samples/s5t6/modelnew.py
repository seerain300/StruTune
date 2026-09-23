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


# Triton vector kernel: compute per-row std from sum_rows and sumsq_rows
@triton.jit
def std_rows_kernel(sum_ptr, sumsq_ptr, std_ptr, S: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    mean = tl.load(sum_ptr + pid) / H
    var = tl.load(sumsq_ptr + pid) / H - mean * mean
    # sqrt in Triton
    std = tl.sqrt(var)
    tl.store(std_ptr + pid, std)


# Triton scalar kernel: compute inverse standard normal CDF (Abramowitz-Stegun approximation)
@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    p = tl.load(p_ptr)
    # Constants
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

    # Piecewise computation
    # Lower tail
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid
    # Upper tail
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))

    # Select based on p
    # p in (0,1); handle edges via selection
    select_low = p < p_low
    select_high = p > (1.0 - p_low)
    z = tl.where(select_low, z_low, tl.where(select_high, z_up, z_mid))
    tl.store(z_ptr, z)


# Triton vector kernel: compute per-row thresholds from std and scalar z
@triton.jit
def compute_thresholds_kernel(std_ptr, mean_ptr, z_scalar_ptr, thresholds_ptr, S: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    std = tl.load(std_ptr + pid)
    mean = tl.load(mean_ptr + pid)
    z = tl.load(z_scalar_ptr)  # scalar
    thresh = mean + std * z
    tl.store(thresholds_ptr + pid, thresh)


# Triton kernel: elementwise gating y = max(0, x - threshold), with threshold vector of length S
@triton.jit
def gate_relu_kernel(X_ptr, thresholds_ptr, Out_ptr, S: tl.constexpr, L: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 2D launch: axis0 over batch*seq, axis1 over feature tiles
    pid_row = tl.program_id(axis=0)
    pid_tile = tl.program_id(axis=1)
    row_start = pid_row * H
    feature_start = pid_tile * BLOCK_SIZE
    idx = feature_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < H
    x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
    thresh = tl.load(thresholds_ptr + pid_row)
    y = x - thresh
    # clamp to [0, inf): max(0, y)
    y = tl.where(y > 0.0, y, 0.0)
    tl.store(Out_ptr + row_start + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float):
    # Ensure CUDA
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    # Shape
    B, L, H = inputs.shape
    S = B * L

    # Flatten to [S, H], contiguous
    X = inputs.contiguous().view(S, H)
    X = X.to(torch.float32)

    # Allocate intermediates
    sum_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    std_rows = torch.empty(S, dtype=torch.float32, device=X.device)
    mean_rows = (sum_rows / H)  # placeholder to have mean available; compute std first

    # Reduction kernels: one program per row
    BLOCK_SIZE = 1024
    grid_reduce = (S,)
    row_sum_kernel[grid_reduce](X, sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute std per row in Triton
    std_rows_kernel[(S,)](sum_rows, sumsq_rows, std_rows, S, H)

    # Compute z = _ndtri(target_sparsity) in Triton (scalar)
    sp_scalar = torch.empty((), dtype=torch.float32, device=X.device)  # dummy for signature; we pass value via pointer
    # We need to pass a 1-element tensor holding target_sparsity; Triton will read it.
    sp_tensor = torch.empty(1, dtype=torch.float32, device=X.device)
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=X.device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)

    # Compute per-row thresholds on device using Triton
    thresholds = torch.empty(S, dtype=torch.float32, device=X.device)
    compute_thresholds_kernel[(S,)](std_rows, (sum_rows / H), z_scalar, thresholds, S)

    # Output tensor
    out = torch.empty_like(X, dtype=torch.float32, device=X.device)

    # Elementwise gating kernel: 2D grid over rows and feature tiles
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, thresholds, out, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original return
    out = out.view(B, L, H)
    out = out.to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)