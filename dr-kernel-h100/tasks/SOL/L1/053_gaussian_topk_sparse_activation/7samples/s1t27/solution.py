import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and population std over last dim N.
# X: [B, S, N] contiguous, but we index via b, s, offs and flatten addressing.
@triton.jit
def row_stats_kernel(
    X_ptr,        # *f32
    MEAN_ptr,     # *f32 [rows]
    STD_ptr,      # *f32 [rows]
    rows: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)  # 0..rows-1
    b = row_id // S
    s = row_id % S

    sum_val = 0.0
    sum_sq = 0.0

    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = b * (S * N) + s * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Triton kernel: scalar ndtri(p) using Abramowitz & Stegun 5.2.23 approximation.
@triton.jit
def ndtri_kernel(
    P_ptr,    # *f32, length 1
    Q_ptr,    # *f32, length 1
    # constants below are inlined
    p_low: tl.constexpr,
    p_high: tl.constexpr,
    a1: tl.constexpr, a2: tl.constexpr, a3: tl.constexpr, a4: tl.constexpr, a5: tl.constexpr, a6: tl.constexpr,
    b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr, c4: tl.constexpr, c5: tl.constexpr, c6: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr, d4: tl.constexpr,
):
    # load p
    p = tl.load(P_ptr)
    q = tl.zeros((), dtype=tl.float32)

    # lower region
    mask_low = p < p_low
    if mask_low:
        # compute ndtri for lower region
        z = tl.sqrt(-2.0 * tl.log(p))
        poly = ((c1 * z + c2) * z + c3) * z + c4
        poly = (poly * z + c5) * z + c6
        t = poly
        u = ((d1 * z + d2) * z + d3) * z + d4
        q = t / (u * z + 1.0)

    # central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        poly_mid = ((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4
        poly_mid = (poly_mid * r_mid + a5) * r_mid + a6
        poly_mid = poly_mid * q_mid
        denom_mid = ((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4
        denom_mid = (denom_mid * r_mid + b5) * r_mid + 1.0
        q = poly_mid / denom_mid

    # upper region
    mask_high = p > p_high
    if mask_high:
        # compute ndtri for upper region
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = ((c1 * z + c2) * z + c3) * z + c4
        poly = (poly * z + c5) * z + c6
        t = poly
        u = ((d1 * z + d2) * z + d3) * z + d4
        q = - (t / (u * z + 1.0))

    tl.store(Q_ptr, q)


# Triton kernel: compute per-row threshold = mean + std * q
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,     # *f32 [rows]
    STD_ptr,      # *f32 [rows]
    q,            # scalar f32
    THRESH_ptr,   # *f32 [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    thresh = mean + std * q
    tl.store(THRESH_ptr + row_id, thresh)


# Triton kernel: elementwise ReLU(x - threshold[row]) over rows, N in chunks.
@triton.jit
def relu_threshold_kernel(
    X_ptr,        # *f32, [B, S, N] contiguous
    THRESH_ptr,   # *f32 [rows]
    OUT_ptr,      # *f32, [rows*N] contiguous
    rows: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    # get threshold for this row
    thr = tl.load(THRESH_ptr + row_id)

    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = b * (S * N) + s * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # ReLU: max(x - thr, 0)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + row_id * N + offs, y, mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
    1) Compute per-row mean and std in Triton (fp32).
    2) Compute ndtri(target_sparsity) in Triton scalar kernel.
    3) Compute per-row threshold in Triton (fp32).
    4) Apply ReLU(x - threshold[row]) elementwise in Triton (fp32).
    5) Return in bf16.
    """
    assert inputs.dim() == 3, "inputs must be 3D [B, S, N]"
    B, S, N = inputs.shape
    rows = B * S

    # Convert to fp32 and ensure contiguous for Triton
    x_f32 = inputs.to(torch.float32).contiguous()

    # 1) Compute mean and std per row
    mean = torch.empty(rows, device=inputs.device, dtype=torch.float32)
    std = torch.empty(rows, device=inputs.device, dtype=torch.float32)
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x_f32, mean, std,
        rows=rows, S=S, N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute ndtri(target_sparsity) in Triton
    p = torch.full((1,), float(target_sparsity), device=inputs.device, dtype=torch.float32)
    q = torch.empty(1, device=inputs.device, dtype=torch.float32)
    # constants from A&S 5.2.23
    p_low = 0.02425
    p_high = 1.0 - p_low
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
    ndtri_kernel[(1,)](
        p, q,
        p_low, p_high,
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5,
        c1, c2, c3, c4, c5, c6,
        d1, d2, d3, d4,
    )

    # 3) Compute per-row threshold
    threshold = torch.empty(rows, device=inputs.device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](
        mean, std, q[0], threshold, rows
    )

    # 4) Apply ReLU elementwise
    OUT = torch.empty(rows * N, device=inputs.device, dtype=torch.float32)
    relu_threshold_kernel[grid_stats](
        x_f32, threshold, OUT,
        rows=rows, S=S, N=N,
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
