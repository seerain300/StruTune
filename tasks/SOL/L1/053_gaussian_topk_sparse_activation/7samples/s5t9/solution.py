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


# Triton kernel: compute per-row std from sums and sumsq
@triton.jit
def std_rows_kernel(Sum_ptr, Sumsq_ptr, Std_ptr, S: tl.constexpr, H: tl.constexpr):
    # Compute mean and std per row
    for i in range(0, S):
        sum_i = tl.load(Sum_ptr + i)
        sumsq_i = tl.load(Sumsq_ptr + i)
        mean = sum_i / H
        var = sumsq_i / H - mean * mean
        std = tl.sqrt(var)  # Triton handles sqrt
        tl.store(Std_ptr + i, std)


# Triton scalar kernel: inverse standard normal CDF (Abramowitz-Stegun 7.1.26)
@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    # p_ptr is a 1-element tensor on device
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

    # Piecewise approximation
    if p < 0.02425:
        # lower region
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > 0.97575:
        # upper region
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # central region
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Store scalar result
    tl.store(z_ptr, z)


# Triton kernel: compute per-row thresholds = mean + std * z_scalar
@triton.jit
def compute_thresholds_kernel(Mean_ptr, Std_ptr, Z_ptr, Thresholds_ptr, S: tl.constexpr):
    # Z_ptr is 1-element tensor on device; load scalar
    z_scalar = tl.load(Z_ptr)
    for i in range(0, S):
        mean_i = tl.load(Mean_ptr + i)
        std_i = tl.load(Std_ptr + i)
        thresh_i = mean_i + std_i * z_scalar
        tl.store(Thresholds_ptr + i, thresh_i)


# Triton kernel: elementwise gating y = max(0, x - threshold), broadcasting thresholds along features
@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S: tl.constexpr, L: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(0)  # batch index
    tile = tl.program_id(1)  # feature tile index
    row = b * L + tile
    if row >= S:
        return
    row_start = row * H
    off = tile * BLOCK_SIZE
    idx = off + tl.arange(0, BLOCK_SIZE)
    mask = idx < H
    x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
    thresh = tl.load(Thresholds_ptr + row)
    y = x - thresh
    y = tl.where(y > 0.0, y, 0.0)  # ReLU
    tl.store(Out_ptr + row_start + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA and contiguous, flatten to [S, H]
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs_f32 = inputs.to(torch.float32).contiguous()
    B, L, H = inputs_f32.shape
    S = B * L

    # View as [S, H]
    X = inputs_f32.view(S, H)

    # 1) Row sum and sum of squares
    sum_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    BLOCK_SIZE = 1024  # tuneable; works for H up to 16K+
    grid_reduce = (S,)
    row_sum_kernel[grid_reduce](X, sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Per-row std
    std_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    std_rows_kernel[(S,)](sum_rows, sumsq_rows, std_rows, S, H)

    # 3) Compute z = _ndtri(target_sparsity) on device via Triton scalar kernel
    sp_tensor = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)  # single program handles scalar

    # 4) Compute per-row thresholds
    thresholds = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    compute_thresholds_kernel[(S,)](sum_rows / H, std_rows, z_scalar, thresholds, S)

    # 5) Elementwise gating
    out = torch.empty(S, H, dtype=torch.float32, device=inputs_f32.device)
    grid_gate = (B, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, thresholds, out, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 6) Reshape and cast to bfloat16 to match original behavior
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
