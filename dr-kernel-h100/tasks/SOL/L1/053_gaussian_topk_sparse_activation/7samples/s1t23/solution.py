import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N for a 2D contiguous [rows, N].
# X_ptr: *f32, [rows, N], contiguous row-major
# MEAN_ptr: *f32, [rows]
# STD_ptr: *f32, [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32, [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows,            # int
    N,               # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0

    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)  # clamp small negatives due to rounding
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar ndtri(target_sparsity) via Abramowitz & Stegun 5.2.23
@triton.jit
def ndtri_kernel(
    p_ptr,           # *f32, [1]
    q_ptr,           # *f32, [1]
):
    p = tl.load(p_ptr)

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

    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    y_mid = poly_mid * q_mid / den_mid

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select appropriate result
    y = tl.where(mask_low, y_low, 0.0) + tl.where(mask_mid, y_mid, 0.0) + tl.where(mask_high, y_high, 0.0)
    tl.store(q_ptr, y)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * multiplier, store to [rows]
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    q_ptr,           # *f32, [1] (scalar multiplier)
    THRESH_ptr,      # *f32, [rows]
    rows,            # int
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(q_ptr)  # scalar
    thr = mean + std * q
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]) on 2D [rows, N], write fp32
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, [rows, N]
    THRESH_ptr,      # *f32, [rows]
    OUT_ptr,         # *f32, [rows*N]
    rows,            # int
    N,               # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # process the entire row in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(THRESH_ptr + row_id)
        y = tl.maximum(x - thr, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes per-row threshold using mean/std over the last dim, then ReLU(input - threshold[row]).
    Returns tensor in bfloat16 to match the original behavior.
    """
    if target_sparsity == 0.0:
        return inputs

    # Make input contiguous along last dim and compute in fp32 for stability
    x = inputs
    # Ensure CUDA and contiguous along the last dimension for Triton
    if not x.is_cuda:
        raise RuntimeError("ModelNew expects CUDA tensors.")
    x_f32 = x.to(torch.float32).contiguous()

    # Shapes
    B, S, N = x_f32.shape
    rows = B * S

    # 1) Compute per-row mean and std over last dim (fp32), contiguous [rows, N]
    x_2d = x_f32.view(rows, N)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)

    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_2d, mean, std,
        rows=rows, N=N,
        BLOCK=2048,  # tuned for large N
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
        x_2d, threshold, OUT,
        rows=rows, N=N,
        BLOCK=2048,
        num_warps=4,
        num_stages=2,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input with shape [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single input tensor.")
        x = args[0]
        return run(x, 0.001)  # target_sparsity is fixed to 0.001 as in the original


def run(*args):
    return ModelNew()(*args)
