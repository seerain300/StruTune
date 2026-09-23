import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std across last dim N for x of shape [B, S, N].
# We launch one program per (i = B*S) row and operate directly on 3D tensor pointers.
@triton.jit
def row_stats_3d_kernel(
    X_ptr,          # *f32, x (converted to f32), shape [B, S, N]
    MEAN_ptr,       # *f32, shape [B*S]
    STD_ptr,        # *f32, shape [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)  # row index over B*S
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        # Compute linear index for this row in the [B, S, N] tensor
        # idx = ((i // S) * N) + (i % S) * N + offs
        b = i // S
        s = i % S
        idx = (b * S + s) * N + offs  # but since we need [B, S, N], better to use b,s separately
        # More robust 3D addressing: linear index = b * (S * N) + s * N + offs
        idx = b * (S * N) + s * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + i, mean)
    tl.store(STD_ptr + i, std)


# Kernel 2: compute ndtri(target_sparsity) using Abramowitz & Stegun 5.2.23 approximation.
# p is a 1-element device tensor; q returns 1-element device tensor with the result.
@triton.jit
def ndtri_kernel(p_ptr, q_ptr,):
    # Load p (scalar)
    p = tl.load(p_ptr)  # float32
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

    # Region masks
    p_low = 0.02425
    p_high = 1.0 - p_low
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    t_low = ((c1 * q_low + c2) * q_low + c3) * q_low + c4
    t_low = (t_low * q_low + c5) * q_low + c6
    u_low = ((d1 * q_low + d2) * q_low + d3) * q_low + d4
    nd_low = t_low / (u_low * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = ((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4
    poly_mid = (poly_mid * r_mid + a5) * r_mid + a6
    poly_mid = poly_mid * q_mid
    denom_mid = ((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4
    denom_mid = (denom_mid * r_mid + b5) * r_mid + 1.0
    nd_mid = poly_mid / denom_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    t_high = ((c1 * q_high + c2) * q_high + c3) * q_high + c4
    t_high = (t_high * q_high + c5) * q_high + c6
    u_high = ((d1 * q_high + d2) * q_high + d3) * q_high + d4
    nd_high = - (t_high / (u_high * q_high + 1.0))

    # Select result
    q_val = tl.where(mask_low, nd_low, 0.0)
    q_val = tl.where(mask_mid, nd_mid, q_val)
    q_val = tl.where(mask_high, nd_high, q_val)
    tl.store(q_ptr, q_val)


# Kernel 3: compute per-row threshold = mean + std * multiplier (scalar q)
@triton.jit
def threshold_3d_kernel(
    MEAN_ptr,       # *f32, [B*S]
    STD_ptr,        # *f32, [B*S]
    Q_ptr,          # *f32, [1]
    THRESH_ptr,     # *f32, [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    rows: tl.constexpr,  # rows = B * S
):
    i = tl.program_id(0)
    mean = tl.load(MEAN_ptr + i)
    std = tl.load(STD_ptr + i)
    q = tl.load(Q_ptr)  # scalar
    thr = mean + std * q
    tl.store(THRESH_ptr + i, thr)


# Kernel 4: elementwise ReLU(x - threshold[i]) across last dim N for each row i.
@triton.jit
def relu_3d_kernel(
    X_ptr,          # *f32, x (converted to f32), shape [B, S, N]
    THRESH_ptr,     # *f32, per-row threshold, shape [B*S]
    OUT_ptr,        # *f32, output, shape [B, S, N]
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)  # row index over B*S
    thresh = tl.load(THRESH_ptr + i)
    # Process last dimension in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        b = i // S
        s = i % S
        idx = b * (S * N) + s * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - thresh, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation:
    - Compute per-row mean and std across last dim.
    - Compute adaptive cutoff threshold = mean + std * ndtri(target_sparsity).
    - Apply ReLU(input - threshold[row]) to create sparse activations.
    Returns output in bf16 (matches original behavior).
    """
    # Ensure we work on CUDA device with Triton; fallback to original if not available
    if inputs.device.type != 'cuda':
        # Fallback to original PyTorch implementation if not on CUDA
        # Convert to float32 for stats, compute mean/std, threshold, ReLU, return bf16
        inputs_f32 = inputs.to(torch.float32)
        mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
        std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
        q = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
        cutoff = mean + std * q
        out = F.relu(inputs_f32 - cutoff)
        return out.to(torch.bfloat16)

    # Convert input to fp32 for reduction accuracy; ensure contiguous
    x = inputs.to(torch.float32).contiguous()
    B, S, N = x.shape
    rows = B * S

    # 1) Compute per-row mean and std in Triton
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    grid_stats = (rows,)
    row_stats_3d_kernel[grid_stats](x, mean, std, B, S, N, BLOCK=1024, num_warps=8)

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton
    p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
    q = torch.empty(1, device=x.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute per-row threshold = mean + std * q
    threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
    threshold_3d_kernel[grid_stats](mean, std, q, threshold, B, S, rows)

    # 4) Elementwise ReLU against per-row threshold, write to fp32 OUT
    OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
    relu_3d_kernel[grid_stats](x, threshold, OUT, B, S, N, BLOCK=1024, num_warps=4)

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


# Optional: original reference _ndtri for CPU fallback (not used in Triton forward)
def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    """
    # Constants for the approximation
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

    # This is a CPU fallback only; Triton forward uses the Triton kernel above.
    result = torch.zeros_like(p)
    # Lower region
    mask_low = p < p_low
    q_low = torch.sqrt(-2.0 * torch.log(p))
    t_low = ((c1 * q_low + c2) * q_low + c3) * q_low + c4
    t_low = (t_low * q_low + c5) * q_low + c6
    u_low = ((d1 * q_low + d2) * q_low + d3) * q_low + d4
    nd_low = t_low / (u_low * q_low + 1.0)
    result = torch.where(mask_low, nd_low, result)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = ((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4
    poly_mid = (poly_mid * r_mid + a5) * r_mid + a6
    poly_mid = poly_mid * q_mid
    denom_mid = ((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4
    denom_mid = (denom_mid * r_mid + b5) * r_mid + 1.0
    nd_mid = poly_mid / denom_mid
    result = torch.where(mask_mid, nd_mid, result)

    # Upper region
    mask_high = p > p_high
    q_high = torch.sqrt(-2.0 * torch.log(1.0 - p))
    t_high = ((c1 * q_high + c2) * q_high + c3) * q_high + c4
    t_high = (t_high * q_high + c5) * q_high + c6
    u_high = ((d1 * q_high + d2) * q_high + d3) * q_high + d4
    nd_high = - (t_high / (u_high * q_high + 1.0))
    result = torch.where(mask_high, nd_high, result)
    return result


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
