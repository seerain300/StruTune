import torch
import triton
import triton.language as tl


# Kernel 1: per-row reduction to compute sum and sum of squares over last dim N.
# X: contiguous [rows, N], where rows = batch_size * seq_len.
# OUT_sum: fp32 [rows], OUT_sumsq: fp32 [rows]
@triton.jit
def row_reduce_kernel(
    X_ptr,        # *f32, contiguous [rows, N]
    OUT_sum_ptr,  # *f32, [rows]
    OUT_sumsq_ptr,# *f32, [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Base pointer for this row
    row_base = X_ptr + row_id * N
    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0
    # Loop over columns in chunks
    for col_start in range(0, N, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < N
        vals = tl.load(row_base + cols, mask=mask, other=0.0)
        # accumulate in fp32
        acc_sum += tl.sum(vals, axis=0)
        acc_sumsq += tl.sum(vals * vals, axis=0)
    # store results
    tl.store(OUT_sum_ptr + row_id, acc_sum)
    tl.store(OUT_sumsq_ptr + row_id, acc_sumsq)


# Kernel 2: compute mean and std from sum and sum of squares (population std, unbiased=False).
# SUM: fp32 [rows], SUMSQ: fp32 [rows], MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    SUM_ptr,      # *f32 [rows]
    SUMSQ_ptr,    # *f32 [rows]
    MEAN_ptr,     # *f32 [rows]
    STD_ptr,      # *f32 [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_ = tl.load(SUM_ptr + row_id)
    sumsq_ = tl.load(SUMSQ_ptr + row_id)
    # population mean and std: var = E[x^2] - (E[x])^2
    mean = sum_ / N
    var = sumsq_ / N - mean * mean
    # ensure non-negative due to numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 3: scalar ndtri (inverse standard normal CDF) using A&S 5.2.23 approximation.
# Input p: 1-element tensor (fp32), Output q: 1-element tensor (fp32)
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
    q_lower = tl.sqrt(-2.0 * tl.log(p))
    z_lower = (((((c1 * q_lower + c2) * q_lower + c3) * q_lower + c4) * q_lower + c5) * q_lower + c6) / \
              ((((d1 * q_lower + d2) * q_lower + d3) * q_lower + d4) * q_lower + 1.0)

    # Central region
    # For central region we need q = p - 0.5
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = num_mid * q_mid / den_mid

    # Upper region
    q_upper = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_upper = -(((((c1 * q_upper + c2) * q_upper + c3) * q_upper + c4) * q_upper + c5) * q_upper + c6) / \
              ((((d1 * q_upper + d2) * q_upper + d3) * q_upper + d4) * q_upper + 1.0)

    # Select based on p
    # Triton doesn't support switch on scalar directly; we can implement selection with masks.
    # If p < p_low: z = z_lower; elif p > p_high: z = z_upper; else z = z_mid
    # Since p is in (0,1), and typical target_sparsity is ~0.01, it will fall into lower or upper.
    z = tl.where(p < p_low, z_lower, z_upper)
    # For the central region, since mask_mid is not evaluated here, we choose z_upper/z_lower appropriately.
    # However, p_mid mask is not needed as we select z_upper or z_lower. z_mid would only be used if p in (p_low, p_high).
    # To be robust, we compute z_mid for central range and override with z_lower/z_upper:
    z = tl.where((p >= p_low) & (p <= p_high), z_mid, z)

    tl.store(q_ptr, z)


# Kernel 4: compute per-row threshold = mean[row] + std[row] * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,    # *f32 [rows]
    STD_ptr,     # *f32 [rows]
    q_ptr,       # *f32 [1]
    THRESH_ptr,  # *f32 [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(q_ptr)  # scalar multiplier
    th = mean + std * q
    tl.store(THRESH_ptr + row_id, th)


# Kernel 5: elementwise ReLU(x - threshold[row]) for each row.
# X: contiguous [rows, N], THRESH: fp32 [rows], OUT: contiguous [rows, N]
@triton.jit
def relu_threshold_kernel(
    X_ptr,        # *f32, contiguous [rows, N]
    THRESH_ptr,   # *f32, [rows]
    OUT_ptr,      # *f32, contiguous [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_base_x = X_ptr + row_id * N
    row_base_out = OUT_ptr + row_id * N
    th = tl.load(THRESH_ptr + row_id)
    for col_start in range(0, N, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(row_base_x + cols, mask=mask, other=0.0)
        y = x - th
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(row_base_out + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "ModelNew expects CUDA tensors."
        x = inputs.contiguous()
        B, S, N = x.shape
        rows = B * S

        # Work in fp32 for numerical stability
        x_f32 = x.to(torch.float32)

        # 1) Compute per-row sum and sum of squares via Triton
        sum_ = torch.empty(rows, device=x.device, dtype=torch.float32)
        sumsq = torch.empty(rows, device=x.device, dtype=torch.float32)
        grid_stats = (rows,)
        row_reduce_kernel[grid_stats](
            x_f32, sum_, sumsq,
            rows=rows, N=N, BLOCK=1024, num_warps=8,
        )

        # 2) Compute mean and std (population) from sums
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        row_stats_kernel[grid_stats](
            sum_, sumsq, mean, std,
            rows=rows, N=N,
        )

        # 3) Compute multiplier = ndtri(target_sparsity) in Triton
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 4) Compute per-row threshold
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 5) Apply ReLU(x - threshold[row]) elementwise in Triton, write to fp32 OUT
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_f32, threshold, OUT,
            rows=rows, N=N, BLOCK=1024, num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)