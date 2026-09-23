import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, Sum_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Sum accumulator
    acc = 0.0
    # Loop over features in tiles
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row-major [S, H], row_id is the row index
        ptr = X_ptr + row_id * H + offs
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    # Write out the row sum
    tl.store(Sum_ptr + row_id, acc)


@triton.jit
def row_sumsq_kernel(X_ptr, Sumsq_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    acc = 0.0
    for start in range(0, H, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        ptr = X_ptr + row_id * H + offs
        vals = tl.load(ptr, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(Sumsq_ptr + row_id, acc)


@triton.jit
def std_rows_kernel(Sum_ptr, Sumsq_ptr, Std_ptr, S: tl.int32, H: tl.int32):
    # One program per row
    row_id = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row_id)
    sumsq_val = tl.load(Sumsq_ptr + row_id)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    std = tl.sqrt(var)
    tl.store(Std_ptr + row_id, std)


@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr, p_low=0.02425):
    # Load p (scalar)
    p = tl.load(p_ptr)
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

    # Regions
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / denom
    elif p > (1.0 - p_low):
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / denom
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly / denom
    tl.store(z_ptr, z)


@triton.jit
def compute_thresholds_kernel(mean_ptr, std_ptr, z_scalar_ptr, thresholds_ptr, S: tl.int32):
    # One program per row
    row_id = tl.program_id(0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_scalar_ptr)  # scalar
    thr = mean + std * z
    tl.store(thresholds_ptr + row_id, thr)


@triton.jit
def gate_relu_kernel(X_ptr, thresholds_ptr, Out_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    # 2D grid: (row, tile)
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H
    x_ptr = X_ptr + row_id * H + offs
    thr_ptr = thresholds_ptr + row_id
    thr = tl.load(thr_ptr)  # scalar threshold for this row
    x = tl.load(x_ptr, mask=mask, other=0.0)
    y = x - thr
    y = tl.maximum(y, 0.0)  # ReLU
    out_ptr = Out_ptr + row_id * H + offs
    tl.store(out_ptr, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Ensure CUDA and contiguous
    if inputs.device.type != 'cuda':
        inputs = inputs.to('cuda')
    inputs_f32 = inputs.contiguous().to(torch.float32)
    B, L, H = inputs_f32.shape
    S = B * L

    # Flatten to [S, H]
    X = inputs_f32.view(S, H)

    # 1) Compute per-row sum and sumsq
    sum_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    grid_sum = (S,)
    row_sum_kernel[grid_sum](X, sum_rows, S, H, BLOCK_SIZE=1024, num_warps=4)
    row_sumsq_kernel[grid_sum](X, sumsq_rows, S, H, BLOCK_SIZE=1024, num_warps=4)

    # 2) Compute per-row std
    std_rows = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    std_rows_kernel[grid_sum](sum_rows, sumsq_rows, std_rows, S, H)

    # 3) Compute z = _ndtri(target_sparsity) on device via Triton scalar kernel
    sp_tensor = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    sp_tensor[0] = float(target_sparsity)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_scalar)

    # 4) Compute per-row thresholds
    thresholds = torch.empty(S, dtype=torch.float32, device=inputs_f32.device)
    compute_thresholds_kernel[(S,)](sum_rows / H, std_rows, z_scalar, thresholds, S)

    # 5) Elementwise gating
    out = torch.empty(S, H, dtype=torch.float32, device=inputs_f32.device)
    grid_gate = (S, triton.cdiv(H, 1024))
    gate_relu_kernel[grid_gate](X, thresholds, out, S, H, BLOCK_SIZE=1024, num_warps=4)

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