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
    rows,            # int32
    N,               # int32
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Bounds check for safety (grid should be rows, but guard anyway)
    if row_id >= rows:
        return
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: compute inverse standard normal CDF (quantile) for a single scalar p.
# p: [1] (device tensor), q: [1] (device tensor output)
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # Load p
    p = tl.load(p_ptr)  # scalar
    # Constants for Abramowitz & Stegun 5.2.23 approximation
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

    # If p out of (0,1), clamp to keep numerical stability
    # Triton scalar math
    if p <= 0.0:
        p = 0.0
    if p >= 1.0:
        p = 1.0

    # Lower region
    if p < p_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        q_val = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
                ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
    else:
        # Central region
        if p > p_high:
            region = 'upper'
        else:
            region = 'central'

        if region == 'central':
            q = p - 0.5
            r = q * q
            q_val = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                    (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        else:
            # Upper region
            z = tl.sqrt(-2.0 * tl.log(1.0 - p))
            q_val = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
                    ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)

    tl.store(q_ptr, q_val)


# Kernel 3: compute per-row threshold: mean[row] + std[row] * multiplier
@triton.jit
def threshold_vec_kernel(MEAN_ptr, STD_ptr, MULTIPLIER_ptr, THRESH_ptr, rows):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    multiplier = tl.load(MULTIPLIER_ptr)  # scalar
    thresh = mean + std * multiplier
    tl.store(THRESH_ptr + row_id, thresh)


# Kernel 4: elementwise ReLU(x - threshold[row]) over 2D grid [rows, N]
# X_ptr: *f32, THRESH_ptr: *f32 [rows], OUT_ptr: *f32
@triton.jit
def relu_threshold_2d_kernel(X_ptr, THRESH_ptr, OUT_ptr,
                             rows, N,
                             BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    if row_id >= rows:
        return
    col_start = col_block_id * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N
    # Load mean threshold for this row
    thresh = tl.load(THRESH_ptr + row_id)
    # Compute linear indices
    idx = row_id * N + offs
    x = tl.load(X_ptr + idx, mask=mask, other=0.0)
    y = x - thresh
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, N], compute ReLU(x - threshold[row]) where
        threshold[row] = mean(row) + std(row) * ndtri(target_sparsity)
        Returns tensor in bf16, matching original behavior.
        """
        # Ensure CUDA and float32 compute
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x_f32 = x.to(torch.float32).contiguous()
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Flatten to [rows, N] and compute mean, std in Triton
        x_flat = x_f32.view(rows, N)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_flat,
            mean,
            std,
            rows=rows,
            N=N,
            BLOCK=1024,
            num_warps=8,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
        p = torch.full((1,), self.target_sparsity, device=x_f32.device, dtype=torch.float32)
        q = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)
        multiplier = q  # 1-element tensor on device

        # 3) Compute threshold per row (fp32) in Triton
        threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, multiplier, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise in Triton (2D grid)
        OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)
        BLOCK_E = 256
        grid = (rows, triton.cdiv(N, BLOCK_E))
        relu_threshold_2d_kernel[grid](
            x_flat,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=BLOCK_E,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
