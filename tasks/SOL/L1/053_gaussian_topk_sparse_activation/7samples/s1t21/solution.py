import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] contiguous in row-major order; rows = batch_size * seq_len.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows,            # int
    N,               # int
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
    # population std (unbiased=False): sqrt(E[x^2] - mean^2)
    var = sum_sq / N - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store per-row statistics
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar ndtri(target_sparsity) via Abramowitz & Stegun 5.2.23
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

    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly = (((((c1*q_low + c2)*q_low + c3)*q_low + c4)*q_low + c5)*q_low + c6)
    denom = (((((d1*q_low + d2)*q_low + d3)*q_low + d4)*q_low + 1.0))
    q_low = poly / denom

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly2 = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
    denom2 = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    q_mid = poly2 * q_mid / denom2

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly3 = (((((c1*q_high + c2)*q_high + c3)*q_high + c4)*q_high + c5)*q_high + c6)
    denom3 = (((((d1*q_high + d2)*q_high + d3)*q_high + d4)*q_high + 1.0))
    q_high = -poly3 / denom3

    # Select q based on mask
    q_val = tl.zeros((), dtype=tl.float32)
    q_val = tl.where(mask_low, q_low, q_val)
    q_val = tl.where(mask_mid, q_mid, q_val)
    q_val = tl.where(mask_high, q_high, q_val)

    tl.store(q_ptr, q_val)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * q
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    q,               # f32 scalar
    THRESH_ptr,      # *f32, [rows]
    rows,            # int
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    threshold = mean + std * q
    tl.store(THRESH_ptr + row_id, threshold)


# Kernel 4: elementwise ReLU(x - threshold[row]) on [rows, N], write to OUT
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    THRESH_ptr,      # *f32, [rows]
    OUT_ptr,         # *f32, [rows*N]
    rows,            # int
    N,               # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # For each row, iterate columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        threshold = tl.load(THRESH_ptr + row_id)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Convert to fp32 for numerical stability; assume inputs are [B, S, N]
        B, S, N = inputs.shape
        rows = B * S

        # Ensure contiguous and cast to fp32 for Triton kernels
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # 1) Compute per-row mean and std using Triton reduction
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        x_2d = x_f32.view(rows, N)
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_2d,
            mean,
            std,
            rows=rows,
            N=N,
            BLOCK=2048,
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
            x_2d,
            threshold,
            OUT,
            rows=rows, N=N,
            BLOCK=2048,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
