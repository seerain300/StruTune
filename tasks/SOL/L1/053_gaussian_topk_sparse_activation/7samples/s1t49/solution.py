import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N], contiguous in row-major order. MEAN: fp32[rows], STD: fp32[rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32
    MEAN_ptr,        # *f32
    STD_ptr,         # *f32
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0

    for col_start in range(0, N, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < N
        idx = row_id * N + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute inverse normal CDF (ndtri) for a given probability p (scalar).
# Uses Abramowitz & Stegun 5.2.23 approximation.
@triton.jit
def ndtri_kernel(
    p_ptr,   # *f32, shape [1]
    q_ptr,   # *f32, shape [1]
):
    p = tl.load(p_ptr)
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

    # Piecewise approximation
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        q = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
    elif p > p_high:
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        q = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
    else:
        q = p - 0.5
        r = q * q
        q = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * q (scalar q).
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,   # *f32, [rows]
    STD_ptr,    # *f32, [rows]
    q,          # f32 scalar
    THRESH_ptr, # *f32, [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    thr = mean + std * q
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]) over [rows, N], write into OUT2D
@triton.jit
def relu_threshold_2d_kernel(
    X_ptr,      # *f32, [rows, N] (row-major)
    THRESH_ptr, # *f32, [rows]
    OUT_ptr,    # *f32, [rows, N] (row-major)
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N

    mean = tl.load(THRESH_ptr + row_id)
    thr = mean  # THRESH_ptr actually holds thresholds in our host code; if needed, use mean+std*q here

    # We need threshold per row. Our host passes threshold per row already in THRESH_ptr.
    # Compute per-row threshold = mean[row] + std[row] * q? Wait: THRESH_ptr should be threshold vector.
    # But threshold_vec_kernel already computed mean + std * q for each row. Here we load it.
    thr = tl.load(THRESH_ptr + row_id)

    idx = row_id * N + cols
    x = tl.load(X_ptr + idx, mask=mask, other=0.0)
    y = x - thr
    y = tl.maximum(y, 0.0)
    tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        assert inputs.is_cuda, "ModelNew requires CUDA tensor input."
        inputs = inputs.contiguous()
        B, S, N = inputs.shape
        rows = B * S

        # Convert to fp32 for stable statistics computation
        x = inputs.to(torch.float32)

        # 1) Flatten to [rows, N] for row-wise reduction
        x_flat = x.view(rows * N)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # Launch row-wise reduction
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
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32) in Triton
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[(rows,)](mean, std, q.item(), threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton (2D grid)
        # Allocate output 2D buffer in fp32
        OUT2D = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        BLOCK_E = 1024
        grid = (rows, triton.cdiv(N, BLOCK_E))
        relu_threshold_2d_kernel[grid](
            x.view(rows, N),
            threshold,
            OUT2D,
            rows=rows,
            N=N,
            BLOCK=BLOCK_E,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT2D.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
