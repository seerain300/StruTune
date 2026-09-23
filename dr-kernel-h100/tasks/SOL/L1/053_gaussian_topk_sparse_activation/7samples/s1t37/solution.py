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
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute inverse standard normal CDF for a scalar p in (0,1) using A&S 5.2.23.
# Uses three regions: lower, central, upper.
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # p_ptr: 1-element tensor, float32
    # q_ptr: 1-element tensor, float32 output
    p = tl.load(p_ptr)
    # constants
    p_low = 0.02425
    p_high = 1.0 - p_low

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

    # lower region
    mask_low = p < p_low
    q_low = ((c1 * tl.sqrt(-2.0 * tl.log(p))) +
             c2) * tl.sqrt(-2.0 * tl.log(p)) + c3
    q_low = (q_low * tl.sqrt(-2.0 * tl.log(p)) + c4) * tl.sqrt(-2.0 * tl.log(p)) + c5
    q_low = (q_low * tl.sqrt(-2.0 * tl.log(p)) + c6) * tl.sqrt(-2.0 * tl.log(p))
    q_low = q_low / ((d1 * tl.sqrt(-2.0 * tl.log(p))) +
                     d2) * tl.sqrt(-2.0 * tl.log(p)) + d3
    q_low = q_low / ((d1 * tl.sqrt(-2.0 * tl.log(p))) + d4)

    # central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = ((a1 * (p - 0.5) * (p - 0.5)) +
             a2) * (p - 0.5) * (p - 0.5) + a3
    q_mid = (q_mid * (p - 0.5) * (p - 0.5) + a4) * (p - 0.5) * (p - 0.5) + a5
    q_mid = (q_mid * (p - 0.5) * (p - 0.5) + a6) * (p - 0.5)

    denom = ((b1 * (p - 0.5) * (p - 0.5)) +
             b2) * (p - 0.5) * (p - 0.5) + b3
    denom = (denom * (p - 0.5) * (p - 0.5) + b4) * (p - 0.5) * (p - 0.5) + b5
    q_mid = q_mid / denom

    # upper region (reflect)
    mask_high = p > p_high
    q_upper = -((c1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) +
                c2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c3
    q_upper = (q_upper * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c4) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c5
    q_upper = (q_upper * tl.sqrt(-2.0 * tl.log(1.0 - p)) + c6) * tl.sqrt(-2.0 * tl.log(1.0 - p))
    q_upper = q_upper / ((d1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) +
                         d2) * tl.sqrt(-2.0 * tl.log(1.0 - p)) + d3
    q_upper = q_upper / ((d1 * tl.sqrt(-2.0 * tl.log(1.0 - p))) + d4)

    # select appropriate branch
    q = tl.zeros((), dtype=tl.float32)
    q = tl.where(mask_low, q_low, q)
    q = tl.where(mask_mid, q_mid, q)
    q = tl.where(mask_high, q_upper, q)

    # write result to q_ptr
    tl.store(q_ptr, q)


# Kernel: compute per-row threshold = mean[row] + std[row] * multiplier (scalar)
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, multiplier_ptr, threshold_ptr, rows: tl.constexpr):
    # multiplier_ptr points to a 1-element tensor
    multiplier = tl.load(multiplier_ptr)
    for row in range(0, rows):
        m = tl.load(mean_ptr + row)
        s = tl.load(std_ptr + row)
        t = m + s * multiplier
        tl.store(threshold_ptr + row, t)


# Kernel: elementwise ReLU against per-row threshold using 2D grid
@triton.jit
def relu_threshold_kernel_2d(
    X2D_ptr,         # *f32, [rows, N]
    threshold_ptr,   # *f32, [1]  (we pass a 1-element tensor, but index per row in Python)
    OUT_ptr,         # *f32, [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    col_start = col_block * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    # load threshold for this row (threshold is a 1-element tensor; we index per row at host side)
    # Note: In this kernel, we assume threshold_ptr points to per-row threshold via host-side mapping.
    # Triton requires contiguous pointers; we instead pass threshold via host-side indexing.
    # To keep it simple and correct, we pass threshold as a 1D tensor [rows] from host and use it directly.
    # Here, we re-launch relu_threshold_kernel_vec that takes threshold per row; this 2D version is not needed.

    # We'll replace this 2D kernel with a per-row kernel below to avoid any confusion.
    pass


# Simpler, per-row elementwise kernel (preferred for robustness and performance):
@triton.jit
def relu_threshold_kernel_vec(
    X_ptr,           # *f32, [rows*N] linearized
    threshold_ptr,   # *f32, [rows]
    OUT_ptr,         # *f32, [rows*N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(threshold_ptr + row_id)
        y = tl.maximum(x - thr, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
    1) Per-row mean and std across feature dim (N)
    2) threshold[row] = mean[row] + std[row] * ndtri(target_sparsity)
    3) Apply ReLU(input - threshold[row]) to create sparse activations

    Returns:
        Sparsified tensor of same shape as input, dtype bf16.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and convert to fp32 for numerics
    x = inputs.contiguous()
    x_f32 = x.to(torch.float32)
    B, S, N = x_f32.shape
    rows = B * S

    # 1) Compute per-row mean and std
    mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

    # Flatten X to [rows, N] for row_stats_kernel addressing
    X2D = x_f32.view(rows, N)
    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        X2D, mean, std,
        rows=rows, N=N, BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
    p = torch.empty(1, device=x_f32.device, dtype=torch.float32)  # 1-element tensor
    p.fill_(float(target_sparsity))  # fill on device
    q = torch.empty(1, device=x_f32.device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)  # q[0] = ndtri(target_sparsity)

    # 3) Compute threshold per row (fp32): threshold[row] = mean[row] + std[row] * q[0]
    threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    threshold_vec_kernel[(rows,)](mean, std, q, threshold, rows)

    # 4) Apply ReLU(x - threshold[row]) elementwise in Triton, per-row kernel
    OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
    BLOCK = 1024
    grid = (rows,)
    relu_threshold_kernel_vec[grid](
        X2D, threshold, OUT,
        rows=rows, N=N, BLOCK=BLOCK,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect inputs as in the original: run(inputs, target_sparsity)
        if len(args) != 2:
            raise RuntimeError("ModelNew expects (inputs, target_sparsity) as arguments.")
        inputs, target_sparsity = args
        return run(inputs, target_sparsity)


def run(*args):
    return ModelNew()(*args)
