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
    var = sum_sq / N - mean * mean
    # Population std (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar inverse standard normal CDF using A&S 5.2.23
# Inputs: p (1-element tensor) in (0, 1). Output: q (1-element tensor)
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    p0 = tl.load(p_ptr)
    # Constants for approximation
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
    # q = sqrt(-2*log(p))
    logp = tl.log(p0)
    q = tl.sqrt(-2.0 * logp)
    res_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    cond_low = p0 < p_low

    # Central region
    # z = p - 0.5, t = z*z
    z = p0 - 0.5
    t = z * z
    res_mid = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6) * z / \
              (((((b1 * t + b2) * t + b3) * t + b4) * t + b5) * t + 1.0)
    cond_mid = (p0 >= p_low) & (p0 <= p_high)

    # Upper region
    log1m = tl.log(1.0 - p0)
    q2 = tl.sqrt(-2.0 * log1m)
    res_up = -(((((c1 * q2 + c2) * q2 + c3) * q2 + c4) * q2 + c5) * q2 + c6) / \
             ((((d1 * q2 + d2) * q2 + d3) * q2 + d4) * q2 + 1.0)
    cond_up = p0 > p_high

    # Select region based on p0
    # Triton supports scalar where; combine masks
    res = tl.where(cond_low, res_low, 0.0) + tl.where(cond_mid, res_mid, 0.0) + tl.where(cond_up, res_up, 0.0)
    tl.store(q_ptr, res)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * multiplier
# MEAN: [rows], STD: [rows], multiplier: scalar (1-element tensor), OUT: [rows]
@triton.jit
def threshold_vec_kernel(MEAN_ptr, STD_ptr, multiplier_ptr, OUT_ptr, rows: tl.constexpr):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    mult = tl.load(multiplier_ptr)
    thresh = mean + std * mult
    tl.store(OUT_ptr + row_id, thresh)


# Kernel 4: elementwise ReLU(x - threshold[row]), X is [rows, N], threshold is [rows]
# OUT is a flat [rows*N] buffer; we write into it and return .view(B, S, N)
@triton.jit
def relu_threshold_kernel(X_ptr, THRESH_ptr, OUT_ptr, rows: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    thresh = tl.load(THRESH_ptr + row_id)
    # Iterate across columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean/std in Triton.
        - Compute ndtri(target_sparsity) in Triton.
        - Compute per-row threshold = mean + std * multiplier in Triton.
        - Apply elementwise ReLU(x - threshold[row]) in Triton and return bf16.
        """
        # Ensure fp32 compute for statistics and activation
        x_f32 = x.to(torch.float32)

        # Flatten to [rows, N]
        B, S, N = x_f32.shape
        rows = B * S
        # Make a contiguous [rows, N] view for Triton kernels
        x_flat = x_f32.view(rows * N)
        # We will treat X as a contiguous [rows, N] by linear indexing inside kernels

        # 1) Compute per-row mean and std in Triton
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

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
        p = torch.full((1,), float(target_sparsity), device=x_f32.device, dtype=torch.float32)
        q = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32) in Triton
        threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
        OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_flat,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=2048,    # larger block reduces loop iterations
            num_warps=8,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
