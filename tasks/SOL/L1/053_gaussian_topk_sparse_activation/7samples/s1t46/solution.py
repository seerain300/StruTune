import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: flattened [rows, N], rows = B * S, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
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
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Scalar kernel: compute inverse standard normal CDF (quantile) for p using A&S 5.2.23
@triton.jit
def ndtri_kernel(
    p_ptr,   # *f32, shape [1]
    q_ptr,   # *f32, shape [1]
):
    # Load probability
    p = tl.load(p_ptr)
    # Constants for A&S approximation
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
    q_lower = (((((c1 * tl.sqrt(-2.0 * tl.log(p))) + c2) * tl.sqrt(-2.0 * tl.log(p)) + c3) * tl.sqrt(-2.0 * tl.log(p)) +
                c4) * tl.sqrt(-2.0 * tl.log(p)) + c5) * tl.sqrt(-2.0 * tl.log(p)) + c6
    den_lower = (((((d1 * tl.sqrt(-2.0 * tl.log(p))) + d2) * tl.sqrt(-2.0 * tl.log(p)) + d3) * tl.sqrt(-2.0 * tl.log(p)) +
                  d4) * tl.sqrt(-2.0 * tl.log(p)) + 1.0)
    q_lower = -q_lower / den_lower  # negative root

    # Central region
    q_central = (((((a1 * (p - 0.5)) + a2) * (p - 0.5) + a3) * (p - 0.5) + a4) * (p - 0.5) + a5) * (p - 0.5) + a6
    den_central = (((((b1 * (p - 0.5)) + b2) * (p - 0.5) + b3) * (p - 0.5) + b4) * (p - 0.5) + b5)
    q_central = q_central / den_central

    # Upper region
    mask_high = p > p_high
    q_upper = (((((c1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) + c2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c3) * tl.sqrt(-2.0 * tl.log(1.0 - p)) +
                c4) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c5) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c6
    den_upper = (((((d1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) + d2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + d3) * tl.sqrt(-2.0 * tl.log(1.0 - p)) +
                  d4) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + 1.0)
    q_upper = q_upper / den_upper

    q = tl.where(mask_low, q_lower, q_central)
    q = tl.where(mask_high, q_upper, q)
    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold in fp32: threshold[row] = mean[row] + std[row] * q
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,    # *f32 [rows]
    STD_ptr,     # *f32 [rows]
    q,           # scalar f32
    THRESH_ptr,  # *f32 [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    thr = mean + std * q
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]) for each row; X is flattened [rows, N]
@triton.jit
def relu_threshold_kernel(
    X_ptr,        # *f32
    THRESH_ptr,   # *f32 [rows]
    OUT_ptr,      # *f32
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Compute base linear index for this row
    base = row_id * N
    thr = tl.load(THRESH_ptr + row_id)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = base + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - thr, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        - Compute per-row mean and std (fp32).
        - Compute adaptive cutoff threshold = mean + std * ndtri(target_sparsity).
        - Apply ReLU(x - threshold[row]) elementwise.
        Returns bf16 tensor matching original behavior.
        """
        if target_sparsity == 0.0:
            # No sparsity
            return x.to(torch.bfloat16)

        # Ensure fp32 for statistics and math
        x_f32 = x.to(torch.float32).contiguous()

        B, S, N = x_f32.shape
        rows = B * S

        # 1) Flatten [B, S, N] -> [rows, N] and compute mean, std per row in fp32
        x_flat = x_f32.view(rows * N)
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
            BLOCK=2048,
            num_warps=8,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
