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
    # iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # reduce within the chunk
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute inverse standard normal CDF for a scalar p in (0,1).
# Uses Abramowitz & Stegun 5.2.23 approximation; returns a 1-element fp32 tensor via pointer.
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # p_ptr: 1-element tensor, float32
    # q_ptr: 1-element tensor, float32 output
    p = tl.load(p_ptr)
    # lower region
    p_low = 0.02425
    p_high = 1.0 - p_low
    mask_low = p < p_low
    mask_high = p > p_high

    # central region constants (Abramowitz & Stegun 5.2.23)
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

    # evaluate central region polynomial for z = sqrt(-2*log(p))
    z = tl.sqrt(-2.0 * tl.log(p))
    z2 = z * z
    # Horner form for numerator and denominator
    num = (((((a1 * z2 + a2) * z2 + a3) * z2 + a4) * z2 + a5) * z2 + a6) * z
    den = (((((b1 * z2 + b2) * z2 + b3) * z2 + b4) * z2 + b5) * z2 + 1.0)
    approx = num / den

    # lower and upper tails
    # lower: x ~ (((((c1*t + c2)*t + c3)*t + c4)*t + c5)*t + c6) / (((((d1*t + d2)*t + d3)*t + d4)*t + 1)*t)
    # upper: x = -approx (since z is sqrt(-2*log(1-p)) ~ sqrt(-2*log(p)) but sign flips due to 1-p)
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

    t_low = 1.0 - p
    poly_c = (((((c1 * t_low + c2) * t_low + c3) * t_low + c4) * t_low + c5) * t_low + c6)
    poly_d = (((((d1 * t_low + d2) * t_low + d3) * t_low + d4) * t_low) + 1.0)
    x_low = poly_c / poly_d

    t_high = p  # for upper tail, use p
    poly_c_high = (((((c1 * t_high + c2) * t_high + c3) * t_high + c4) * t_high + c5) * t_high + c6)
    poly_d_high = (((((d1 * t_high + d2) * t_high + d3) * t_high + d4) * t_high) + 1.0)
    x_high = -poly_c_high / poly_d_high

    # select result based on region
    # For exact p=0.5, central region is fine; otherwise one of the branches dominates.
    # The approximation is sufficiently accurate for our needs.
    # We can simply choose the central approximation for general p, as it performs well across (0,1).
    # However, to be robust, select region logic:
    # If p in [p_low, p_high], use approx; else use low branch for p<p_low, high branch for p>p_high.
    # Here we implement selection via tl.where-like behavior using masks:
    result = approx
    if mask_low:
        result = x_low
    elif mask_high:
        result = x_high

    tl.store(q_ptr, result)


# Kernel 3: compute per-row threshold = mean + std * multiplier
# MEAN: fp32 [rows], STD: fp32 [rows], multiplier: scalar from q (1-element tensor)
# THRESH: fp32 [rows]
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, p_ptr, THRESH_ptr, rows: tl.constexpr):
    # Load multiplier scalar
    m = tl.load(p_ptr)  # 1-element tensor, scalar load
    for row in range(0, rows):
        mean = tl.load(mean_ptr + row)
        std = tl.load(std_ptr + row)
        thresh = mean + std * m
        tl.store(THRESH_ptr + row, thresh)


# Kernel 4: elementwise ReLU(x - threshold[row]) over [rows, N] using 2D grid
# X2D: *f32, [rows, N]
# THRESH: *f32, [rows] (we pass pointer; kernel will load per row)
# OUT: *f32, [rows, N]
@triton.jit
def relu_threshold_kernel_2d(X2D_ptr, THRESH_ptr, OUT_ptr, rows: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    col_start = col_block * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    # load threshold for this row
    thr = tl.load(THRESH_ptr + row_id)

    # compute linear indices
    idx = row_id * N + offs
    x = tl.load(X2D_ptr + idx, mask=mask, other=0.0)
    y = x - thr
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Computes per-row mean and std in Triton (fp32)
    - Computes ndtri(target_sparsity) in Triton (scalar)
    - Computes per-row threshold in Triton
    - Applies elementwise ReLU(x - threshold[row]) in Triton
    - Returns output in bf16
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure fp32 for numerical stability
    x = inputs.to(torch.float32)

    # Flatten to [rows, N] where N = last dimension
    N = x.shape[-1]
    rows = x.numel() // N
    x2d = x.view(rows, N)

    # Allocate output buffer in fp32 for computation
    out_fp32 = torch.empty(rows * N, device=x.device, dtype=torch.float32)

    # 1) Compute per-row mean and std
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x2d,
        mean,
        std,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton
    p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
    q = torch.empty(1, device=x.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute per-row threshold in Triton
    threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](
        mean, std, q, threshold, rows
    )

    # 4) Apply ReLU(x - threshold[row]) using 2D grid
    BLOCK = 2048
    grid = (rows, triton.cdiv(N, BLOCK))
    relu_threshold_kernel_2d[grid](
        x2d, threshold, out_fp32,
        rows=rows, N=N, BLOCK=BLOCK,
        num_warps=8,
    )

    # Return in bf16 to match original behavior
    return out_fp32.view(*x.shape).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) != 2:
            raise RuntimeError("ModelNew expects (inputs, target_sparsity) as arguments.")
        inputs, target_sparsity = args
        return run(inputs, target_sparsity)