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
        # Compute linear indices for this row
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Reduce within the chunk
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    # population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)
    # Store results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Triton scalar kernel: compute inverse normal CDF (quantile) for probability p
# Uses Abramowitz & Stegun 5.2.23 approximation.
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # Load probability
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
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        y = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6) / \
            ((((d1*z + d2)*z + d3)*z + d4)*z + 1.0)
        q = -y
    # Central region
    elif p <= p_high:
        z = p - 0.5
        r = z * z
        y = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * z / \
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        q = y
    # Upper region
    else:
        z = tl.sqrt(-2.0 * tl.log(1.0 - p))
        y = (((((c1*z + c2)*z + c3)*z + c4)*z + c5)*z + c6) / \
            ((((d1*z + d2)*z + d3)*z + d4)*z + 1.0)
        q = z - y
    tl.store(q_ptr, q)


# Kernel 3: compute per-row threshold = mean + std * multiplier
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, multiplier, threshold_ptr, rows: tl.constexpr):
    row_id = tl.program_id(0)
    m = tl.load(mean_ptr + row_id)
    s = tl.load(std_ptr + row_id)
    q = tl.load(multiplier)  # scalar
    t = m + s * q
    tl.store(threshold_ptr + row_id, t)


# Kernel 4: elementwise ReLU(x - threshold[row]) per row
@triton.jit
def relu_threshold_kernel(X_ptr, threshold_ptr, OUT_ptr, rows: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    base = row_id * N
    # Loop over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = base + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        t = tl.load(threshold_ptr + row_id)  # scalar per row
        y = x - t
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


# Entry point model
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure input is contiguous and on CUDA
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous()
        # Compute in fp32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Flatten leading dims into rows
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Compute per-row mean and std in Triton
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
        p = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        q = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        p.fill_(float(target_sparsity))
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32) in Triton
        threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton
        # Flatten X to [rows, N] for linear addressing, write to OUT
        X_lin = x_f32.view(rows * N)  # linearization for pointer arithmetic
        OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            X_lin,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=2048,    # larger block reduces loop iterations
            num_warps=8,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)