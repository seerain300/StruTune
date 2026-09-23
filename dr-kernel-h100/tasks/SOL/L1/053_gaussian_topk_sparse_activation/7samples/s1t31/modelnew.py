import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] where rows = batch_size * seq_len, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Scalar Triton kernel: compute ndtri(p) for p in (0,1) using A&S approximation (5.2.23)
# p: 1-element tensor on device, q: 1-element output tensor on device
@triton.jit
def ndtri_kernel(p, q):
    # Abramowitz & Stegun coefficients
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

    # constants
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Load p (1-element)
    p0 = tl.load(p)
    # Lower region
    mask_low = p0 < p_low
    q0 = tl.zeros_like(p0)
    if mask_low:
        u = tl.sqrt(-2.0 * tl.log(p0))
        poly = (((((c1 * u + c2) * u + c3) * u + c4) * u + c5) * u + c6)
        poly2 = (((((d1 * u + d2) * u + d3) * u + d4) * u + 1.0))
        q0 = poly / poly2

    # Central region
    mask_mid = (p0 >= p_low) & (p0 <= p_high)
    if mask_mid:
        # evaluate central approximation: (p - 0.5) is small for typical inputs
        z = p0 - 0.5
        r = z * z
        poly3 = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly4 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        q0 = poly3 * z / poly4

    # Upper region
    mask_high = p0 > p_high
    if mask_high:
        u = tl.sqrt(-2.0 * tl.log(1.0 - p0))
        poly = (((((c1 * u + c2) * u + c3) * u + c4) * u + c5) * u + c6)
        poly2 = (((((d1 * u + d2) * u + d3) * u + d4) * u + 1.0))
        q0 = -poly / poly2

    # Store result
    tl.store(q, q0)


# Kernel 2: compute threshold per row in fp32: threshold[row] = mean[row] + std[row] * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    MULTIPLIER_ptr,  # *f32, [1] (scalar)
    THRESH_ptr,      # *f32, [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    mult = tl.load(MULTIPLIER_ptr)  # scalar load
    thr = mean + std * mult
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 3: elementwise ReLU(x - threshold[row]), X is [rows, N] flattened, THRESH is [rows]
# OUT is fp32, linearized as [rows * N]
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, [rows * N] flattened
    THRESH_ptr,      # *f32, [rows]
    OUT_ptr,         # *f32, [rows * N] flattened
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Each program handles one row; loop over columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(THRESH_ptr + row_id)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-row threshold using inverse standard normal CDF, then
    outputs max(0, x - threshold[row]).
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure inputs are contiguous
    x = inputs.contiguous()
    B, S, N = x.shape
    rows = B * S

    # Compute in fp32 for numerical stability
    x_f32 = x.to(torch.float32)

    # 1) Compute per-row mean and std over last dim N
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_f32.view(rows * N),
        mean,
        std,
        rows=rows,
        N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
    p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
    q = torch.empty(1, device=x.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute threshold per row (fp32) in Triton
    threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

    # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
    OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
    relu_threshold_kernel[grid_stats](
        x_f32, threshold, OUT,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        assert len(args) == 1, "run expects a single input tensor"
        return run(*args)