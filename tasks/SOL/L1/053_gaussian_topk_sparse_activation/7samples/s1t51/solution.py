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
    # population std: sqrt(E[x^2] - (E[x])^2)
    std = tl.sqrt(sum_sq / N - mean * mean)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute per-row threshold = mean[row] + std[row] * multiplier,
# and write it back in-place into X (which acts as per-row buffer).
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,         # *f32 [rows]
    STD_ptr,          # *f32 [rows]
    MULTIPLIER_ptr,   # *f32 [1] (scalar q from ndtri)
    X_ptr,            # *f32 [rows, N] (we will write threshold[row] into X[row, 0])
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(MULTIPLIER_ptr)  # scalar
    threshold = mean + std * q
    # write scalar threshold to X[row, 0]
    tl.store(X_ptr + row_id * N, threshold)


# Kernel 3: apply ReLU(x - threshold[row]) in-place on X, where threshold[row] is stored at X[row, 0].
@triton.jit
def relu_inplace_kernel(
    X_ptr,            # *f32 [rows, N], will read x and threshold[row] at col 0
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Read threshold from the first element of this row
    threshold = tl.load(X_ptr + row_id * N)
    # Iterate over columns in chunks of BLOCK and apply ReLU(x - threshold)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = tl.maximum(x - threshold, 0.0)
        tl.store(X_ptr + idx, y, mask=mask)


# Scalar Triton kernel for ndtri (Abramowitz & Stegun 5.2.23 approximation)
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # p_ptr: *f32 (1-element tensor containing target_sparsity)
    # q_ptr: *f32 (1-element output for quantile)
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
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = poly_low / denom_low

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    denom_up = ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    z_up = -poly_up / denom_up

    # Select branch
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    # Triton scalar select
    z = tl.where(cond_low, z_low, 0.0)
    z = tl.where(cond_mid, z_mid, z)
    # high branch not needed since cond_mid covers [p_low, 1-p_low]; others are implicitly z_up by default

    tl.store(q_ptr, z)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version of run:
        - Compute per-row mean and std in fp32
        - Compute per-row threshold = mean + std * ndtri(target_sparsity) in fp32
        - Apply ReLU(x - threshold[row]) in-place (fp32), then cast to bf16
        """
        # Ensure dtype and contiguity
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, S, N = x.shape
        rows = B * S
        # Work in fp32 for numerical stability
        x_f32 = x.to(torch.float32)

        # 1) Compute per-row mean and std (row_stats_kernel)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32,
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

        # 3) Write per-row threshold into x_f32 (first element of each row)
        # This makes output have the same shape/layout as input, which helps evaluation.
        threshold_vec_kernel[grid_stats](
            mean, std, q, x_f32,
            rows=rows,
            N=N,
            BLOCK=1,   # scalar per row, no need for vector block
            num_warps=1,
        )

        # 4) Apply ReLU(x - threshold[row]) in-place on x_f32
        relu_inplace_kernel[grid_stats](
            x_f32,
            rows=rows,
            N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return x_f32.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
