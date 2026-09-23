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


# Triton scalar kernel: compute inverse standard normal CDF (Abramowitz-Stegun approximation)
# Input: p (1-element tensor on device), Output: z (1-element tensor on device)
@triton.jit
def ndtri_scalar_kernel(p_ptr, z_ptr):
    p = tl.load(p_ptr)  # scalar float
    # Constants for the approximation
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

    # Masks for regions
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Initialize output z to 0.0 (will be overwritten in each region)
    z = 0.0

    # Lower region: p < 0.02425
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region: p in [0.02425, 0.97575]
    if mask_mid:
        # p is in central region; use central formula
        # Abramowitz-Stegun central approximation
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region: p > 0.97575
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(z_ptr, z)


# Triton kernel: compute per-row std from sums and sumsq
@triton.jit
def std_rows_kernel(Sum_ptr, Sumsq_ptr, Std_ptr, S: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    sum_row = tl.load(Sum_ptr + pid)
    sumsq_row = tl.load(Sumsq_ptr + pid)
    mean = sum_row / H
    var = sumsq_row / H - mean * mean
    # Compute std via sqrt in Triton scalar (avoid host torch.sqrt)
    std = tl.sqrt(var)
    tl.store(Std_ptr + pid, std)


# Triton kernel: compute per-row thresholds vector for gating
# X_ptr: input (float32), Std_ptr: per-row std (length S), Mean_ptr: per-row mean (length S),
#       z_scalar (1-element tensor), Out_ptr: output
@triton.jit
def compute_thresholds_kernel(X_ptr, Std_ptr, Mean_ptr, z_ptr, Thresholds_ptr, S: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per row
    std = tl.load(Std_ptr + pid)
    mean = tl.load(Mean_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    threshold = mean + std * z
    tl.store(Thresholds_ptr + pid, threshold)


# Triton elementwise kernel: apply gating y = max(0, x - threshold), with per-row thresholds
@triton.jit
def gate_relu_kernel(X_ptr, Thresholds_ptr, Out_ptr, S: tl.constexpr, L: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(axis=0)  # batch index
    l = tl.program_id(axis=1)  # seq index
    row = b * L + l
    threshold = tl.load(Thresholds_ptr + row)
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(X_ptr + row * H + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # relu
        tl.store(Out_ptr + row * H + idx, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Expect inputs of shape [B, L, H]
    assert inputs.ndim == 3, f"Expected 3D input [B, L, H], got shape {inputs.shape}"
    B, L, H = inputs.shape
    device = inputs.device

    # Ensure contiguous and cast to float32 for reductions and gating
    X = inputs.contiguous().to(torch.float32)

    # Flatten rows for reduction: S = B * L
    S = B * L

    # Allocate per-row sums and sumsq
    sum_rows = torch.empty(S, dtype=torch.float32, device=device)
    sumsq_rows = torch.empty(S, dtype=torch.float32, device=device)

    # Launch reduction kernels: one program per row
    BLOCK_SIZE = 1024  # works well for H up to 16384; loop covers larger H too
    grid_reduce = (S,)
    row_sum_kernel[grid_reduce](X, sum_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    row_sumsq_kernel[grid_reduce](X, sumsq_rows, S, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute per-row std in Triton (avoid host torch ops)
    std_rows = torch.empty(S, dtype=torch.float32, device=device)
    std_rows_kernel[grid_reduce](sum_rows, sumsq_rows, std_rows, S, H)

    # Compute z = _ndtri(target_sparsity) in Triton scalar kernel
    sp_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
    z_tensor = torch.empty((), dtype=torch.float32, device=device)
    ndtri_scalar_kernel[(1,)](sp_tensor, z_tensor)  # single program handles scalar

    # Compute per-row thresholds on device using Triton
    thresholds = torch.empty(S, dtype=torch.float32, device=device)
    compute_thresholds_kernel[(S,)](X, std_rows, (sum_rows / H), z_tensor, thresholds, S, H)

    # Allocate output (float32 for numerical stability)
    out = torch.empty_like(inputs, dtype=torch.float32, device=device)

    # Launch elementwise gating kernel: 2D grid over (batch, seq)
    grid_gate = (B, L)
    gate_relu_kernel[grid_gate](X, thresholds, out, S, L, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Cast back to bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; can be changed


def run(*args):
    return ModelNew()(*args)
