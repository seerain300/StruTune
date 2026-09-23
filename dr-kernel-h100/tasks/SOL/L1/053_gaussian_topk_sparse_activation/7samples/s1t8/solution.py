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
    rows: tl.constexpr,  # number of rows (B*S)
    N: tl.constexpr,     # feature size
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Guard against extra programs
    if row_id >= rows:
        return

    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        # Row-major contiguous: offset = row_id * N + offs
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        # Masked reduction: sum only valid elements
        sum_val += tl.sum(tl.where(mask, x, 0.0))
        sum_sq += tl.sum(tl.where(mask, x * x, 0.0))

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    # std = sqrt(var)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar inverse normal CDF using A&S 5.2.23 approximation.
# Input p: device scalar tensor (1 element), output q: device scalar tensor (1 element).
@triton.jit
def ndtri_kernel(
    p_ptr,           # *f32, shape [1], contains probability
    q_ptr,           # *f32, shape [1], output inverse CDF
):
    p = tl.load(p_ptr)
    # A&S constants
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

    # Masks for regions
    mask_low = p < p_low
    mask_high = p > p_high
    mask_mid = (~mask_low) & (~mask_high)

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    u = p - 0.5
    q_mid = 1.0  # placeholder
    r = u * u
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # handle divide-by-zero is not expected for p in (0.5, 0.5) but keep safe
    q_mid = (poly * u) / poly2

    # Upper region (use 1 - p)
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select based on mask
    q = q_low * mask_low + q_mid * mask_mid + y_up * mask_high
    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold = mean + std * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,    # *f32, [rows]
    STD_ptr,     # *f32, [rows]
    MULTIPLIER_ptr,  # *f32, [1]
    THRESH_ptr,  # *f32, [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    m = tl.load(MULTIPLIER_ptr)  # scalar multiplier from ndtri
    th = mean + std * m
    tl.store(THRESH_ptr + row_id, th)


# Kernel 4: elementwise ReLU(x - threshold[row]) over last dim N
@triton.jit
def relu_threshold_kernel(
    X_ptr,        # *f32, linearized [rows, N]
    THRESH_ptr,   # *f32, [rows], threshold per row
    OUT_ptr,      # *f32, [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    thr = tl.load(THRESH_ptr + row_id)

    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        diff = x - thr
        # ReLU
        y = tl.maximum(diff, 0.0)
        tl.store(OUT_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Assume x is [B, S, N]
        B, S, N = x.shape
        rows = B * S

        # Ensure contiguous and float32 for compute
        x_f32 = x.to(torch.float32).contiguous()

        # 1) Compute per-row mean and std
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32,
            mean,
            std,
            rows=rows,
            N=N,
            BLOCK=1024,   # tuned for throughput
            num_warps=8,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
        p = torch.empty(1, device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        p.fill_(float(target_sparsity))
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32) in Triton
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_f32,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
