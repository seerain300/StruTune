import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: *f32, contiguous [rows, N] where rows = batch_size * seq_len, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32
    MEAN_ptr,        # *f32
    STD_ptr,         # *f32
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over N in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row, mean)
    tl.store(STD_ptr + row, std)


# Kernel 2: inverse standard normal CDF (Abramowitz & Stegun 5.2.23), for a single scalar p
# p_ptr: *f32, 1-element tensor with probability on device
# q_ptr: *f32, 1-element tensor to write result
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    p = tl.load(p_ptr)  # scalar float32
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
    p_high = 1.0 - p_low

    # Region selection
    if p < p_low:
        # Lower region
        z = tl.sqrt(-2.0 * tl.log(p))
        y = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        q = -y
    elif p > p_high:
        # Upper region
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        y = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
        q = y
    else:
        # Central region
        q = p - 0.5
        r = q * q
        y = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * q
# MEAN: *f32 [rows], STD: *f32 [rows], q: scalar f32, THRESH: *f32 [rows]
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,
    STD_ptr,
    q,                # scalar float32
    THRESH_ptr,       # *f32
    rows: tl.constexpr,
):
    row = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row)
    std = tl.load(STD_ptr + row)
    thresh = mean + std * q
    tl.store(THRESH_ptr + row, thresh)


# Kernel 4: elementwise ReLU against per-row threshold, 2D grid over rows and tiles of N
# X_flat: *f32, flattened [rows * N], THRESH: *f32 [rows], OUT: *f32 [rows * N]
@triton.jit
def relu_row_kernel(
    X_flat_ptr,
    THRESH_ptr,
    OUT_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col_start = tile * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N
    # Load per-row threshold
    thresh = tl.load(THRESH_ptr + row)
    base = row * N
    x = tl.load(X_flat_ptr + base + offs, mask=mask, other=0.0)
    y = x - thresh
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(OUT_ptr + base + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Compute per-row mean and std (fp32).
    - Compute ndtri(target_sparsity) in Triton (scalar).
    - Compute per-row threshold = mean + std * multiplier.
    - Apply ReLU(x - threshold[row]) elementwise in Triton.
    - Return in bfloat16.
    """
    # Ensure inputs are contiguous and cast to float32 for compute
    x = inputs.contiguous()
    B, S, N = x.shape
    rows = B * S
    device = x.device

    # Flatten to [rows, N] for kernel addressing
    x_flat = x.view(rows * N).contiguous()

    # Allocate outputs for statistics and thresholds
    mean = torch.empty(rows, device=device, dtype=torch.float32)
    std = torch.empty(rows, device=device, dtype=torch.float32)
    threshold = torch.empty(rows, device=device, dtype=torch.float32)

    # 1) Compute per-row mean and std
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_flat,
        mean,
        std,
        rows=rows,
        N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
    p = torch.full((1,), float(target_sparsity), device=device, dtype=torch.float32)
    q = torch.empty(1, device=device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute threshold per row (fp32)
    threshold_vec_kernel[grid_stats](mean, std, q.item(), threshold, rows)

    # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
    OUT = torch.empty(rows * N, device=device, dtype=torch.float32)
    grid_rows = (rows,)
    grid_tiles = (triton.cdiv(N, 1024),)
    relu_row_kernel[grid_rows + grid_tiles](
        x_flat,
        threshold,
        OUT,
        N=N,
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bfloat16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        return run(inputs, target_sparsity)