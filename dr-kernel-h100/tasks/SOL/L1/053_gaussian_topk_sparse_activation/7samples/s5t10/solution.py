import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over features in chunks
    for start in range(0, H, BLOCK_SIZE):
        idx = start + offs
        mask = idx < H
        # Compute linear index for row-major [S, H]
        ptr = X_ptr + row * H + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    # Store per-row sum
    tl.store(Sum_ptr + row, acc)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, H, BLOCK_SIZE):
        idx = start + offs
        mask = idx < H
        ptr = X_ptr + row * H + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + row, acc)


@triton.jit
def std_rows_kernel(Sum_ptr, Sumsq_ptr, Std_ptr, S, H):
    # One program per row
    row = tl.program_id(0)
    sum_row = tl.load(Sum_ptr + row)
    sumsq_row = tl.load(Sumsq_ptr + row)
    mean = sum_row / H
    var = sumsq_row / H - mean * mean
    # Ensure non-negative due to approximation
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(Std_ptr + row, std)


@triton.jit
def ndtri_scalar_kernel(P_ptr, Z_ptr):
    # Compute inverse normal CDF for p = P_ptr[0] via A&S 7.1.26
    # p_ptr: 1-element tensor; z_ptr: 1-element tensor
    p = tl.load(P_ptr)  # scalar
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

    p_low = 0.02425
    # piecewise evaluation
    # lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    r_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = r_low

    # central region
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # upper region
    mask_high = p > (1.0 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    r_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = -r_high

    # select z based on region
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(Z_ptr, z)


@triton.jit
def compute_thresholds_kernel(Mean_ptr, Std_ptr, Z_scalar_ptr, Thresholds_ptr, S):
    # One program per row
    row = tl.program_id(0)
    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    z = tl.load(Z_scalar_ptr)
    thr = mean + std * z
    tl.store(Thresholds_ptr + row, thr)


@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    # 2D grid: rows x feature tiles
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x_ptr = X_ptr + row * H + offs
    thr_ptr = Thresholds_ptr + row
    thr = tl.load(thr_ptr)
    x = tl.load(x_ptr, mask=mask, other=0.0)
    y = x - thr
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(Out_ptr + row * H + offs, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA and contiguous
    if not inputs.is_cuda:
        raise RuntimeError("ModelNew requires CUDA input tensor")
    inputs_f32 = inputs.contiguous().to(torch.float32)
    B, L, H = inputs_f32.shape
    S = B * L

    # Flatten to [S, H]
    X = inputs_f32.view(S, H)

    # 1) Compute per-row sum
    sum_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    row_sum_kernel[(S,)](X, sum_rows, S, H, BLOCK_SIZE=1024)

    # 2) Compute per-row sum of squares
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    row_sumsq_kernel[(S,)](X, sumsq_rows, S, H, BLOCK_SIZE=1024)

    # 3) Compute per-row std
    std_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    std_rows_kernel[(S,)](sum_rows, sumsq_rows, std_rows, S, H)

    # 4) Compute z = _ndtri(target_sparsity) via Triton scalar kernel
    sp_tensor = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)  # single program handles scalar

    # 5) Compute per-row thresholds
    thresholds = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    mean_rows = sum_rows / H
    compute_thresholds_kernel[(S,)](mean_rows, std_rows, z_scalar, thresholds, S)

    # 6) Elementwise gating y = max(0, x - threshold)
    out = torch.empty(S, H, dtype=torch.float32, device=inputs_f32.device)
    grid_gate = (S, triton.cdiv(H, 1024))
    gate_relu_kernel[grid_gate](X, thresholds, out, S, H, BLOCK_SIZE=1024, num_warps=4)

    # 7) Reshape and cast to bfloat16 to match original behavior
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
