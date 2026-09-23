import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X2D: fp32 [rows, N], contiguous
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel_2d(
    X2D_ptr,         # *f32, pointer to [rows, N] contiguous
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows,            # int
    N,               # int
    stride_row,      # int (elements)
    stride_col,      # int (elements), typically 1
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
        idx = row_id * stride_row + offs * stride_col
        x = tl.load(X2D_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)  # population std (unbiased=False)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute inverse standard normal CDF for p -> q using A&S 5.2.23
# Input p: fp32 scalar tensor (1 element), Output q: fp32 scalar tensor (1 element)
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    p = tl.load(p_ptr)  # scalar
    # Constants for Abramowitz & Stegun 5.2.23
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
    # Note: q is scalar; branch based on p scalar
    # Compute q
    q = 0.0
    if p < p_low:
        u = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1*u + c2)*u + c3)*u + c4)*u + c5)*u + c6)
        denom = (((((d1*u + d2)*u + d3)*u + d4)*u + 1.0))
        q = poly / denom
    elif p > p_high:
        u = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1*u + c2)*u + c3)*u + c4)*u + c5)*u + c6)
        denom = (((((d1*u + d2)*u + d3)*u + d4)*u + 1.0))
        q = -poly / denom
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        denom = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        q = poly / denom

    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold vector (fp32) from mean, std, and q
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, q_ptr, threshold_ptr, rows):
    # q_ptr is a single-element tensor; load scalar q
    q = tl.load(q_ptr)
    for row in range(0, rows):
        m = tl.load(mean_ptr + row)
        s = tl.load(std_ptr + row)
        tl.store(threshold_ptr + row, m + s * q)


# Kernel 4: elementwise ReLU(x - threshold[row]) on 2D [rows, N]
@triton.jit
def relu_threshold_kernel_2d(
    X2D_ptr,          # *f32, [rows, N]
    threshold_ptr,    # *f32, [rows]
    OUT2D_ptr,        # *f32, [rows, N]
    rows,             # int
    N,                # int
    stride_row,       # int
    stride_col,       # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load threshold for this row
    thr = tl.load(threshold_ptr + row_id)
    # Iterate over columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * stride_row + offs * stride_col
        x = tl.load(X2D_ptr + idx, mask=mask, other=0.0)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT2D_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Computes per-row mean and std over last dim.
    - Computes ndtri(target_sparsity) in Triton.
    - Computes per-row threshold: mean + std * ndtri(sparsity).
    - Applies ReLU(x - threshold[row]) elementwise.
    Returns bf16 tensor with shape [batch_size, seq_len, intermediate_size].
    """
    # Early exit: no sparsity
    if target_sparsity == 0.0:
        return inputs

    # Convert to fp32 for stable stats
    x = inputs.to(torch.float32)

    # Shapes
    B, S, N = x.shape
    rows = B * S

    # Ensure 2D contiguous [rows, N]
    x2d = x.contiguous().view(rows, N)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)

    # Launch row-wise reduction
    grid_stats = (rows,)
    row_stats_kernel_2d[grid_stats](
        x2d, mean, std,
        rows, N,
        x2d.stride(0), x2d.stride(1),
        BLOCK=1024,
        num_warps=4,
    )

    # Compute multiplier = ndtri(target_sparsity) in Triton
    p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
    q = torch.empty(1, device=x.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # Compute threshold per row (fp32)
    threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

    # Elementwise ReLU against per-row threshold, write to OUT2d (fp32)
    OUT2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    relu_threshold_kernel_2d[grid_stats](
        x2d, threshold, OUT2d,
        rows, N,
        OUT2d.stride(0), OUT2d.stride(1),
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT2d.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single input tensor [batch, seq, inter]
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor.")
        return run(args[0])


def run(*args):
    return ModelNew()(*args)
