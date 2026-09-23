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
    # guard in case grid > rows
    if row_id >= rows:
        return

    # pointers to this row
    row_base = X_ptr + row_id * N
    # accumulate in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # iterate over N in chunks of BLOCK
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(row_base + offs, mask=mask, other=0.0)
        # sum and sum of squares
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.full((), N, tl.float32)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    # population std
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar inverse normal CDF (Abramowitz & Stegun 5.2.23)
# Input: p (1-element tensor with probability in (0,1))
# Output: q (1-element tensor with inverse CDF)
@triton.jit
def ndtri_kernel(P_ptr, Q_ptr):
    # load probability
    p = tl.load(P_ptr)
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
    q_low = tl.sqrt(-2.0 * tl.log(p))
    approx_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                 ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    result_low = approx_low

    # central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    approx_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                 (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    result_mid = approx_mid

    # upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    approx_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                   ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    result_high = approx_high

    # piecewise selection
    # Triton doesn't have tl.where; build masks and select using tl.load/tl.store or arithmetic trick.
    # Here we compute piecewise via masks since we can't branch per element. Use P_ptr trick:
    # We need to return q; compute as a linear combination of masks, but Triton scalar doesn't support masks.
    # Instead, return result_mid by default, and adjust for regions by comparing p to thresholds.
    # Since this is scalar, implement via simple comparisons and assignments:
    if p < p_low:
        q_val = result_low
    elif p <= p_high:
        q_val = result_mid
    else:
        q_val = result_high

    tl.store(Q_ptr, q_val)


# Kernel 3: compute per-row threshold in fp32
@triton.jit
def threshold_vec_kernel(MEAN_ptr, STD_ptr, q, THRESH_ptr, rows: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    mult = tl.load(q)  # scalar q
    thr = mean + std * mult
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]) per row
@triton.jit
def relu_threshold_kernel(X_ptr, THRESH_ptr, OUT_ptr, rows: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    base_x = X_ptr + row_id * N
    thr = tl.load(THRESH_ptr + row_id)
    base_out = OUT_ptr + row_id * N

    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(base_x + offs, mask=mask, other=0.0)
        diff = x - thr  # scalar thr
        y = tl.maximum(diff, 0.0)
        tl.store(base_out + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        x = x.contiguous()
        B, S, N = x.shape
        rows = B * S

        # Compute in fp32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Allocate mean and std vectors (fp32)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # Launch row-wise reduction
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32, mean, std,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=8,
        )

        # Compute multiplier = ndtri(target_sparsity) in Triton
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # Elementwise ReLU against per-row threshold, write to fp32 OUT
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_f32, threshold, OUT,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
