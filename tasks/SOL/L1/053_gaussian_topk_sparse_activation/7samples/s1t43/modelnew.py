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
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        # Linear indices for this row
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    # Store per-row results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar inverse standard normal CDF (Abramowitz & Stegun 5.2.23)
# Input p is [1] float32, output q is [1] float32
@triton.jit
def ndtri_scalar_kernel(
    p_ptr,    # *f32, shape [1]
    q_ptr,    # *f32, shape [1]
):
    p = tl.load(p_ptr)  # scalar float32
    # Constants for approximation
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
    if p < p_low:
        z = torch.sqrt(-2.0 * torch.log(p))
        # This block is scalar in Triton; z is scalar computed via math functions
        result = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6) / \
                 ((((d1*z + d2)*z + d3)*z + d4)*z + 1.0)
    else:
        # Central region
        if p <= p_high:
            q = p - 0.5
            r = q * q
            result = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
                     (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        else:
            # Upper region
            z = torch.sqrt(-2.0 * torch.log(1.0 - p))
            result = -(((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6) / \
                     ((((d1*z + d2)*z + d3)*z + d4)*z + 1.0)

    tl.store(q_ptr, result)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * q
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,  # *f32, [rows]
    STD_ptr,   # *f32, [rows]
    Q,         # scalar float32
    THRESH_ptr,# *f32, [rows]
    rows: tl.constexpr,
):
    for r in range(0, rows):
        mean = tl.load(MEAN_ptr + r)
        std = tl.load(STD_ptr + r)
        thr = mean + std * Q
        tl.store(THRESH_ptr + r, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]), writing fp32 OUT
@triton.jit
def relu_threshold_row_kernel(
    X_lin_ptr,     # *f32, [rows*N] linearized
    THRESH_ptr,    # *f32, [rows]
    OUT_ptr,       # *f32, [rows*N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    base = row_id * N
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = base + offs
        x = tl.load(X_lin_ptr + idx, mask=mask, other=0.0)
        thr = tl.load(THRESH_ptr + row_id)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Input x shape: [batch_size, seq_len, intermediate_size]
        x = args[0]
        # Ensure contiguous and compute in fp32
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Compute per-row mean and std (fp32)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

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
        target_sparsity = 0.5
        if len(args) > 1 and isinstance(args[1], (float, torch.Tensor)):
            target_sparsity = float(args[1]) if isinstance(args[1], torch.Tensor) else float(args[1])
        p = torch.full((1,), target_sparsity, device=x_f32.device, dtype=torch.float32)
        q = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        ndtri_scalar_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Elementwise ReLU against per-row threshold, linearized
        X_lin = x_f32.view(rows * N)  # linearize for pointer arithmetic
        OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
        relu_threshold_row_kernel[grid_stats](
            X_lin,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=2048,    # larger block reduces loop iterations; safe with masking
            num_warps=8,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)