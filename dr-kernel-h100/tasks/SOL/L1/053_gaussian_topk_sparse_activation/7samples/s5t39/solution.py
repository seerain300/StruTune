import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(X_ptr, S_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    # Accumulate sum across H in tiles
    sum_val = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        start += BLOCK_SIZE
    tl.store(S_ptr + row, sum_val)


@triton.jit
def sumsq_kernel(X_ptr, Ssq_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    sumsq_val = 0.0
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        sumsq_val += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE
    tl.store(Ssq_ptr + row, sumsq_val)


@triton.jit
def mean_std_kernel(Sum_ptr, Ssq_ptr, Mean_ptr, Std_ptr, S, H):
    row = tl.program_id(0)
    sum_val = tl.load(Sum_ptr + row)
    sumsq_val = tl.load(Ssq_ptr + row)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    std = tl.sqrt(var)  # Triton sqrt; no host-side math allowed
    tl.store(Mean_ptr + row, mean)
    tl.store(Std_ptr + row, std)


@triton.jit
def ndtri_kernel(p_ptr, z_ptr):
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF (quantile)
    # p_ptr: 1-element tensor, z_ptr: 1-element output
    p = tl.load(p_ptr)
    # Constants
    p_low = 2.425e-2
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
    mask_high = p > (1.0 - p_low)
    # Select piecewise
    z_val = tl.where(mask_low, z_low, 0.0)
    z_val = tl.where(mask_high, z_high, z_val)
    z_val = tl.where(~mask_low & ~mask_high, z_mid, z_val)
    tl.store(z_ptr, z_val)


@triton.jit
def compute_thresholds_kernel(Mean_ptr, Std_ptr, z_scalar, Thresholds_ptr, S: tl.constexpr):
    row = tl.program_id(0)
    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    thr = mean + std * z_scalar
    tl.store(Thresholds_ptr + row, thr)


@triton.jit
def gate_relu_kernel(X_ptr, Thr_ptr, Out_ptr, S, H, z_scalar, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    # One program per row; process the entire row in tiles
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        thr = tl.load(Thr_ptr + row)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU: max(0, y)
        tl.store(Out_ptr + row * H + offs, y, mask=mask)
        start += BLOCK_SIZE


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation:
    - Compute per-row mean and std (unbiased=False) across last dim.
    - Compute threshold = mean + std * _ndtri(target_sparsity).
    - Apply y = max(0, x - threshold).
    - Return bfloat16 tensor.
    """
    # Move to CUDA if needed and ensure contiguous
    orig_device = inputs.device
    inputs_cuda = inputs
    if not inputs_cuda.is_cuda:
        inputs_cuda = inputs_cuda.cuda()
    inputs_cuda = inputs_cuda.contiguous()

    B, L, H = inputs_cuda.shape
    S = B * L
    X = inputs_cuda.to(torch.float32).view(S, H)

    # Allocate intermediates
    Sum = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Mean = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Std = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Thresholds = torch.empty(S, dtype=torch.float32, device=inputs_cuda.device)
    Out = torch.empty((S, H), dtype=torch.float32, device=inputs_cuda.device)

    # 1) Compute row sums and sums of squares
    BLOCK_SIZE = 1024
    sum_kernel[(S,)](X, Sum, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    sumsq_kernel[(S,)](X, Sumsq, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # 2) Compute per-row mean and std
    mean_std_kernel[(S,)](Sum, Sumsq, Mean, Std, S=S, H=H, num_warps=4)

    # 3) Compute inverse normal CDF for target_sparsity=0.1 in Triton
    p = torch.empty(1, dtype=torch.float32, device=inputs_cuda.device)
    z_scalar = torch.empty(1, dtype=torch.float32, device=inputs_cuda.device)
    p[0] = float(target_sparsity)  # default 0.1
    ndtri_kernel[(1,)](p, z_scalar)  # compute z_scalar

    # 4) Compute per-row thresholds
    compute_thresholds_kernel[(S,)](Mean, Std, z_scalar[0], Thresholds, S=S, num_warps=4)

    # 5) Elementwise gating
    grid_gate = (S, triton.cdiv(H, BLOCK_SIZE))
    gate_relu_kernel[grid_gate](X, Thresholds, Out, S=S, H=H, z_scalar=float(target_sparsity), BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Out.view(B, L, H).to(torch.bfloat16)

    # Move back to original device if input was on CPU
    if orig_device.type != 'cuda':
        out = out.cpu()
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
