import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X2D: [rows, N] in fp32 and contiguous.
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
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar ndtri(p) using Abramowitz & Stegun 5.2.23 approximation.
@triton.jit
def ndtri_kernel(
    p_ptr,   # *f32, 1-element tensor
    q_ptr,   # *f32, 1-element output
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


# Kernel 3: compute per-row threshold = mean[row] + std[row] * q
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32 [rows]
    STD_ptr,         # *f32 [rows]
    q,               # scalar f32
    THRESH_ptr,      # *f32 [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    thr = mean + std * q
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(X - threshold[row]) for each row
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    THRESH_ptr,      # *f32 [rows]
    OUT_ptr,         # *f32, contiguous [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(THRESH_ptr + row_id)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return input as-is (original behavior implicitly; here we return x in bf16)
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous and compute in fp32
        x_f32 = x.to(torch.float32)
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Compute per-row mean and std in Triton over [rows, N]
        x2d = x_f32.contiguous().view(rows, N)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x2d,
            mean,
            std,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 3) Compute per-row threshold (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton, write fp32 OUT2d
        OUT2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x2d,
            threshold,
            OUT2d,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT2d.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
