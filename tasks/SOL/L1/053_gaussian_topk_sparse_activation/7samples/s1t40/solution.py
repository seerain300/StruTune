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
    # population std (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Triton scalar kernel: compute q = ndtri(p) using Abramowitz & Stegun 5.2.23
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # p_ptr: [1] float32, q_ptr: [1] float32
    p = tl.load(p_ptr)
    # constants
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

    # lower region
    # q = sqrt(-2 * log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    t = 1.0 / (1.0 - p)
    poly = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
    denom = (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
    nd_low = poly / denom
    # central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly2 = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly3 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    nd_mid = poly2 * q_mid / poly3
    # upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    t2 = 1.0 / p
    poly4 = (((((c1 * t2 + c2) * t2 + c3) * t2 + c4) * t2 + c5) * t2 + c6)
    denom2 = (((((d1 * t2 + d2) * t2 + d3) * t2 + d4) * t2 + 1.0))
    nd_high = -poly4 / denom2

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # choose appropriate branch; for p in (0,1), one mask is true
    q_val = tl.where(mask_low, q_low, 0.0) + tl.where(mask_mid, nd_mid, 0.0) + tl.where(mask_high, nd_high, 0.0)
    tl.store(q_ptr, q_val)


# Kernel 2: compute threshold per row (fp32)
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, q_ptr, threshold_ptr, rows: tl.constexpr):
    row_id = tl.program_id(0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    q = tl.load(q_ptr)
    thr = mean + std * q
    tl.store(threshold_ptr + row_id, thr)


# Kernel 3: elementwise ReLU against per-row threshold.
# X is [rows, N] laid out linearly as 1D. threshold is [rows].
# Each program handles one row, iterates over N in BLOCK chunks.
@triton.jit
def relu_threshold_kernel_rowwise(
    X_ptr,            # *f32, [rows, N] linearized
    threshold_ptr,    # *f32, [rows]
    OUT_ptr,          # *f32, [rows, N] linearized
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    thr = tl.load(threshold_ptr + row_id)
    base = row_id * N
    # iterate columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        in_idx = base + offs
        x = tl.load(X_ptr + in_idx, mask=mask, other=0.0)
        y = tl.maximum(x - thr, 0.0)  # ReLU
        tl.store(OUT_ptr + in_idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is (original behavior returns same dtype)
        if target_sparsity == 0.0:
            return inputs

        # Compute stats in fp32
        x = inputs
        # Flatten to [rows, N] logically; we'll pass linearized pointers
        B, S, N = x.shape
        rows = B * S

        x_f32 = x.to(torch.float32).contiguous()
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # 1) Row-wise reduction: mean and std
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32, mean, std,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=8,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) in Triton
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Elementwise ReLU against per-row threshold, linearized
        X_lin = x_f32.view(rows * N)  # linearize for pointer arithmetic
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel_rowwise[grid_stats](
            X_lin, threshold, OUT,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
