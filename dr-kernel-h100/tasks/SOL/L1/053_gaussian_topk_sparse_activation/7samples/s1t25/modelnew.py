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


# Kernel 2: compute inverse normal CDF (scalar) using A&S 5.2.23
# Input p: [1] scalar in (0,1), output q: [1] scalar
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # Load p
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

    # Lower region
    mask_low = p < p_low
    q_low = (((((c1 * tl.sqrt(-2.0 * tl.log(p))) + c2) * tl.sqrt(-2.0 * tl.log(p)) + c3) * tl.sqrt(-2.0 * tl.log(p)) + c4) * tl.sqrt(-2.0 * tl.log(p)) + c5) * tl.sqrt(-2.0 * tl.log(p)) + c6
    q_low = -q_low / ((((d1 * tl.sqrt(-2.0 * tl.log(p))) + d2) * tl.sqrt(-2.0 * tl.log(p)) + d3) * tl.sqrt(-2.0 * tl.log(p)) + d4)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = (((((a1 * (p - 0.5) * (p - 0.5) + a2) * (p - 0.5) * (p - 0.5) + a3) * (p - 0.5) * (p - 0.5) + a4) * (p - 0.5) * (p - 0.5) + a5) * (p - 0.5) * (p - 0.5) + a6) * (p - 0.5)
    denom_mid = (((((b1 * (p - 0.5) * (p - 0.5) + b2) * (p - 0.5) * (p - 0.5) + b3) * (p - 0.5) * (p - 0.5) + b4) * (p - 0.5) * (p - 0.5) + b5) * (p - 0.5) * (p - 0.5) + 1.0)
    q_mid = q_mid / denom_mid

    # Upper region
    mask_high = p > p_high
    q_high = (((((c1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) + c2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c3) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c4) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c5) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c6
    q_high = -q_high / ((((d1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) + d2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + d3) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + d4)

    q = tl.where(mask_low, q_low, 0.0)
    q = tl.where(mask_mid, q_mid, q)
    q = tl.where(mask_high, q_high, q)

    tl.store(q_ptr, q)


# Kernel 3: compute threshold per row (fp32): threshold[row] = mean[row] + std[row] * q
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, q_ptr, threshold_ptr, rows: tl.constexpr):
    # q is scalar, broadcast across rows
    q = tl.load(q_ptr)
    for i in range(rows):
        mean_i = tl.load(mean_ptr + i)
        std_i = tl.load(std_ptr + i)
        tl.store(threshold_ptr + i, mean_i + std_i * q)


# Kernel 4: elementwise ReLU(x - threshold[row]) over linearized [rows, N]
@triton.jit
def relu_threshold_kernel(X_lin_ptr, threshold_ptr, OUT_ptr, rows: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * N
    # Load threshold for this row
    thresh = tl.load(threshold_ptr + row_id)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = base + offs
        x = tl.load(X_lin_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - thresh, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.

    1) Compute per-row mean and std over last dimension N.
    2) threshold[row] = mean[row] + std[row] * ndtri(target_sparsity).
    3) Output = ReLU(inputs - threshold[row]).

    Returns:
        Sparsified tensor of same shape as input, dtype bfloat16.
    """
    assert inputs.is_cuda, "inputs must be on CUDA device for Triton kernels"
    # Ensure contiguous and compute in fp32
    x = inputs.contiguous()
    B, S, N = x.shape
    rows = B * S

    # Flatten last two dims to [rows, N]
    x_lin = x.view(rows, N)

    # 1) Reduction to mean and std (fp32)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_lin,
        mean,
        std,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
    p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
    q = torch.empty(1, device=x.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute threshold per row (fp32)
    threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

    # 4) Elementwise ReLU against per-row threshold, write to fp32 OUT
    OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
    relu_threshold_kernel[grid_stats](
        x_lin,
        threshold,
        OUT,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)