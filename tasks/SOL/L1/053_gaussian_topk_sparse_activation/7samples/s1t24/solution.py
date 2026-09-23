import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean and population std over last dim N.
# X_ptr: *f32, contiguous [rows, N]
# MEAN_ptr: *f32, [rows]
# STD_ptr: *f32, [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32
    MEAN_ptr,        # *f32
    STD_ptr,         # *f32
    rows,            # int (runtime)
    N,               # int (runtime)
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Scalar kernel: compute ndtri(target_sparsity) via Abramowitz & Stegun 5.2.23
@triton.jit
def ndtri_kernel(
    p_ptr,           # *f32, [1]
    q_ptr,           # *f32, [1]
):
    # Load p[0]
    p = tl.load(p_ptr)
    # Abramowitz & Stegun constants
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

    # Lower region approximation
    q_lower = tl.sqrt(-2.0 * tl.log(p))
    y_lower = (((((c1 * q_lower + c2) * q_lower + c3) * q_lower + c4) * q_lower + c5) * q_lower + c6) / \
              ((((d1 * q_lower + d2) * q_lower + d3) * q_lower + d4) * q_lower + 1.0)

    # Central region approximation
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    y_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region approximation
    q_upper = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_upper = -(((((c1 * q_upper + c2) * q_upper + c3) * q_upper + c4) * q_upper + c5) * q_upper + c6) / \
              ((((d1 * q_upper + d2) * q_upper + d3) * q_upper + d4) * q_upper + 1.0)

    # Select region based on p
    # Note: Triton does not have branchless select of multiple expressions easily; implement piecewise
    # For simplicity, use piecewise assignment:
    # If p < p_low: q_lower, else if p > p_high: q_upper, else: q_mid
    y = y_lower
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_low, y_lower, y)
    y = tl.where(p > p_high, y_upper, y)

    tl.store(q_ptr, y)


# Kernel: compute per-row threshold vector = mean + std * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    multiplier,      # f32 (runtime scalar)
    threshold_ptr,   # *f32, [rows]
    rows,            # int (runtime)
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    thr = mean + std * multiplier
    tl.store(threshold_ptr + row_id, thr)


# Kernel: elementwise ReLU(x - threshold[row]) over [rows, N], write to OUT
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    threshold_ptr,   # *f32, [rows]
    OUT_ptr,         # *f32, contiguous [rows, N]
    rows,            # int (runtime)
    N,               # int (runtime)
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Each program handles one row; loop over N in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(threshold_ptr + row_id)  # scalar per row
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the Gaussian-based top-k sparse activation.
    - Computes per-row mean and std over last dim.
    - Computes adaptive cutoff threshold: mean + std * ndtri(target_sparsity).
    - Applies ReLU(input - threshold[row]) elementwise, returns bf16.
    """
    # Handle single input tensor
    if len(inputs.shape) != 3:
        raise RuntimeError("ModelNew expects a single input tensor of shape [batch_size, seq_len, intermediate_size].")
    B, S, N = inputs.shape

    # Make a contiguous fp32 2D view [rows, N] for Triton kernels
    x_f32 = inputs.to(torch.float32).contiguous()  # contiguous in memory
    rows = B * S
    x_2d = x_f32.view(rows, N)

    # 1) Compute per-row mean and std (fp32) in Triton
    mean = torch.empty(rows, device=inputs.device, dtype=torch.float32)
    std = torch.empty(rows, device=inputs.device, dtype=torch.float32)

    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_2d,
        mean,
        std,
        rows=rows,
        N=N,
        BLOCK=2048,   # tuned for N up to 12288
        num_warps=8,
        num_stages=2,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
    p = torch.full((1,), float(target_sparsity), device=inputs.device, dtype=torch.float32)
    q = torch.empty(1, device=inputs.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)
    multiplier = q  # 1-element tensor; pass as runtime scalar below

    # 3) Compute threshold per row (fp32) in Triton
    threshold = torch.empty(rows, device=inputs.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](
        mean,
        std,
        float(target_sparsity),  # use host float; Triton will treat it as scalar
        threshold,
        rows=rows,
    )

    # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
    OUT = torch.empty(rows * N, device=inputs.device, dtype=torch.float32)
    relu_threshold_kernel[grid_stats](
        x_2d,
        threshold,
        OUT,
        rows=rows,
        N=N,
        BLOCK=2048,
        num_warps=4,
        num_stages=2,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single input tensor.")
        return run(args[0], 0.001)


def run(*args):
    return ModelNew()(*args)
