import torch
import triton
import triton.language as tl


# Kernel: per-row reduction to compute sum and sum of squares across N.
# X: *f32, pointer to [rows, N] contiguous
# SUM_ptr: *f32, [rows]
# SUMSQ_ptr: *f32, [rows]
@triton.jit
def row_stats_kernel(X_ptr, SUM_ptr, SUMSQ_ptr, rows, N, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    # Accumulate sum and sum of squares across N using chunked loads
    acc_sum = 0.0
    acc_sumsq = 0.0
    # Loop over N in chunks of BLOCK
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        # Compute linear offsets for this row
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
    tl.store(SUM_ptr + row_id, acc_sum)
    tl.store(SUMSQ_ptr + row_id, acc_sumsq)


# Kernel: compute per-row mean and std (population) from sums
# SUM_ptr: *f32, [rows]
# SUMSQ_ptr: *f32, [rows]
# MEAN_ptr: *f32, [rows]
# STD_ptr: *f32, [rows]
@triton.jit
def compute_mean_std_kernel(SUM_ptr, SUMSQ_ptr, MEAN_ptr, STD_ptr, rows, N):
    row_id = tl.program_id(0)
    sum_val = tl.load(SUM_ptr + row_id)
    sumsq_val = tl.load(SUMSQ_ptr + row_id)
    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel: scalar inverse-normal CDF (Abramowitz & Stegun 5.2.23 approximation)
# P: *f32, 1-element tensor (input probability)
# Q: *f32, 1-element tensor (output inverse CDF)
@triton.jit
def ndtri_kernel(P_ptr, Q_ptr):
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

    # Load p
    p = tl.load(P_ptr)  # scalar
    # Handle lower region
    # q = sqrt(-2*log(p)) for p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / denom_low

    # Central region
    # q = p - 0.5
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / denom_high

    # Select region
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_high = p > p_high

    q = tl.where(cond_low, z_low, tl.where(cond_mid, z_mid, z_high))
    tl.store(Q_ptr, q)


# Kernel: compute per-row threshold vector
# MEAN_ptr: *f32, [rows]
# STD_ptr: *f32, [rows]
# Q_scalar: *f32, 1-element tensor containing multiplier
# THRESH_ptr: *f32, [rows]
@triton.jit
def threshold_vec_kernel(MEAN_ptr, STD_ptr, Q_ptr, THRESH_ptr, rows):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(Q_ptr)  # scalar
    tl.store(THRESH_ptr + row_id, mean + std * q)


# Kernel: elementwise ReLU(x - threshold[row]) per row
# X_ptr: *f32, [rows, N] flattened
# THRESH_ptr: *f32, [rows]
# OUT_ptr: *f32, [rows, N] flattened
@triton.jit
def relu_threshold_kernel(X_ptr, THRESH_ptr, OUT_ptr, rows, N, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    # Load scalar threshold for this row
    thr = tl.load(THRESH_ptr + row_id)
    # Process the row in chunks of BLOCK
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure we have 3D input: [batch_size, seq_len, intermediate_size]
        assert x.dim() == 3, "Input must be a 3D tensor [B, S, N]"
        B, S, N = x.shape
        rows = B * S

        # Work in fp32 inside Triton
        x_f32 = x.to(torch.float32).contiguous()

        # 1) Per-row sum and sum of squares (fp32)
        sum_buf = torch.empty(rows, device=x.device, dtype=torch.float32)
        sumsq_buf = torch.empty(rows, device=x.device, dtype=torch.float32)

        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32, sum_buf, sumsq_buf,
            rows=rows, N=N, BLOCK=1024, num_warps=8,
        )

        # 2) Compute mean and std (population) in Triton
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        compute_mean_std_kernel[grid_stats](sum_buf, sumsq_buf, mean, std, rows, N)

        # 3) Compute ndtri(target_sparsity) in Triton (scalar)
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 4) Compute threshold per row (fp32)
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


def run(*args):
    return ModelNew()(*args)
