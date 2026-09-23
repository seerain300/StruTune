import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d_exact(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    Processes the entire row in a single vectorized load (BLOCK_SIZE == N).
    x_ptr: [rows, N]
    mean_ptr/std_ptr: [rows], float32
    N: int32, must equal BLOCK_SIZE
    """
    row_id = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N  # true when BLOCK_SIZE == N
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    sum_val = tl.sum(x)
    sum_sq = tl.sum(x * x)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_scalar(inv_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) for a single scalar p in (0,1).
    Uses Abramowitz & Stegun approximation (formula 26.2.23).
    inv_ptr: [1] float32 output
    p: scalar float32 (passed as runtime arg)
    """
    # Constants for the approximation
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
        q = tl.sqrt(-2.0 * tl.log(p))
        t = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        inv = -t
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        t = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        inv = t
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        t = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        inv = -t

    tl.store(inv_ptr, inv)


@triton.jit
def gate_rows_2d(out_ptr, x_ptr, threshold_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton 2D kernel: apply gating per row over tiles of the last dimension.
    out_ptr: [rows, N]
    x_ptr: [rows, N]
    threshold_ptr: [rows]
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    thresh = tl.load(threshold_ptr + row_id)
    y = x - thresh
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [B, S, N] tensor
        target_sparsity: float in (0, 1)
        Returns bfloat16 tensor with gating y = max(0, x - (mean + std * inv_norm_cdf(target_sparsity))) per row.
        """
        if target_sparsity == 0.0:
            return x

        # Convert input to float32 for numerical stability in Triton
        x_f32 = x.to(torch.float32)

        # Compute shape: rows = B * S, N = last dim
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Compute per-row mean and std in Triton (population std, unbiased=False)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        # Choose BLOCK_SIZE == N to process the entire row exactly
        reduce_mean_std_2d_exact[(rows,)](x_f32, mean, std, N, BLOCK_SIZE=N, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) in Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1, num_warps=1)
        inv_cdf = float(inv_cdf_buf.item())  # scalar

        # 3) Form threshold vector [rows] and apply gating in Triton
        threshold = mean + std * inv_cdf  # [rows]

        # Prepare 2D views [rows, N]
        x_2d = x_f32.view(rows, N)
        out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)

        # Launch 2D gating kernel over tiles of N
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](out_2d, x_2d, threshold, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

        # Reshape back to [B, S, N] and cast to bfloat16 to match original behavior
        out = out_2d.view(B, S, N).to(torch.bfloat16)
        return out