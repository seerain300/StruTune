import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean and population std along last dim (N)
# X is [rows, N] contiguous; outputs MEAN[rows], STD[rows] (fp32).
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32
    MEAN_ptr,        # *f32, length rows
    STD_ptr,         # *f32, length rows
    rows: tl.constexpr,  # int
    N: tl.constexpr,     # int
    BLOCK: tl.constexpr, # int (e.g., 256)
):
    row_id = tl.program_id(0)  # one program per row
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        # Reduce within this chunk
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    # population std: sqrt(E[x^2] - mean^2)
    var = sum_sq / N - mean * mean
    # Ensure non-negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel: compute inverse normal CDF (quantile) for probability p.
# Uses Abramowitz & Stegun 5.2.23 approximation; expects p as 1-element tensor.
@triton.jit
def ndtri_kernel(
    p_ptr,           # *f32, length 1
    q_ptr,           # *f32, length 1
    p_low: tl.constexpr,   # 0.02425
):
    # Load probability
    p = tl.load(p_ptr)  # scalar float32
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

    # Regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q_lower = 0.0
    q_upper = 0.0
    region = 0.0  # will be set to 1 if p < p_low, 2 if p > p_high, else 0

    if p < p_low:
        region = 1.0
        q = tl.sqrt(-2.0 * tl.log(p))
        q_lower = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        region = 2.0
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        q_upper = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        region = 0.0
        # Central region approximation
        z = p - 0.5
        r = z * z
        q_lower = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * z / \
                  (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Select result
    q_out = tl.where(region == 1.0, q_lower, tl.where(region == 2.0, q_upper, q_lower))
    tl.store(q_ptr, q_out)


# Kernel: compute threshold per row: mean + std * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,   # *f32, length rows
    STD_ptr,    # *f32, length rows
    multiplier_ptr,  # *f32, length 1
    THRESH_ptr, # *f32, length rows
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    mult = tl.load(multiplier_ptr)
    thresh = mean + std * mult
    tl.store(THRESH_ptr + row_id, thresh)


# Kernel: elementwise ReLU(x - threshold[row]) over N, per row
@triton.jit
def relu_threshold_kernel(
    X_ptr,           # *f32, [rows, N]
    THRESH_ptr,      # *f32, [rows]
    OUT_ptr,         # *f32, [rows, N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    thresh = tl.load(THRESH_ptr + row_id)
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Flatten leading dims to rows, ensure contiguous CUDA
        assert x.is_cuda, "Input must be on CUDA for Triton."
        x = x.contiguous()
        B, S, N = x.shape
        rows = B * S

        # Compute in fp32 for numerical stability; allocate stats
        # Note: we do not use any torch reductions in forward.
        # Create per-row stats vectors (fp32)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # Launch row-wise stats kernel
        grid_stats = (rows,)
        # Choose BLOCK and num_warps; 256 is a good default for these N sizes
        row_stats_kernel[grid_stats](
            x, mean, std,
            rows=rows,
            N=N,
            BLOCK=256,
            num_warps=4,
        )

        # Compute scalar inverse normal CDF of target_sparsity using Triton
        p = torch.empty(1, device=x.device, dtype=torch.float32)
        p.fill_(float(target_sparsity))  # keep it as a device scalar tensor
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q, p_low=0.02425)  # pass constant, not a tensor arg

        # Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[(rows,)](mean, std, q, threshold, rows)

        # Elementwise ReLU against per-row threshold
        out = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        # Reshape x to [rows, N] logically; we pass linearized pointers and compute row offsets.
        # Launch per-row programs; grid over rows
        relu_threshold_kernel[(rows,)](
            x, threshold, out, rows=rows, N=N, BLOCK=256, num_warps=4
        )

        # Final cast to bf16 to match original behavior
        return out.view(B, S, N).to(torch.bfloat16)