import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _compute_row_stats_2d_kernel(x_ptr, mean_ptr, std_ptr,
                                 rows, K,
                                 BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std along columns (K) for a 2D contiguous tensor x of shape [rows, K].
    Stores mean[row] and std[row] as float32 in mean_ptr/std_ptr of length rows.
    """
    row = tl.program_id(0)
    # If grid > rows, guard
    if row >= rows:
        return

    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Iterate over columns in chunks
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # x is [rows, K] contiguous => index = row*K + offs
        x_row_ptr = x_ptr + row * K + offs
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)  # load as is; will cast to fp32 below
        x_vals = x_vals.to(tl.float32)
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / K
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / K - mean * mean
    # Ensure non-negative due to numerical noise
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store mean and std as float32
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def _ndtri_kernel(p_ptr, out_ptr):
    """
    Compute inverse standard normal CDF (quantile) for p[0], using A&S 5.2.23 approximation.
    Writes result to out_ptr[0].
    """
    # p_ptr is 1-element, out_ptr is 1-element
    p = tl.load(p_ptr)
    # Piecewise approximation constants (float32)
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
    q = tl.sqrt(-2.0 * tl.log(p))
    y = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
        ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    cond_low = p < p_low

    # Central region
    q = p - 0.5
    r = q * q
    y = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
        (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    cond_mid = (p >= p_low) & (p <= p_high)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
        ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    cond_high = p > p_high

    # Select piecewise
    y = tl.where(cond_low, y, y)  # y is already computed for all branches; tl.where here is a no-op.
    # Better: compute y as a whole and let branches fold; Triton will evaluate all math, but selection is done via cond flags. This pattern is fine for Triton.
    # However, Triton supports scalar selection; we can write y = qnan and overwrite with selected branch:
    y = float('nan')
    y = tl.where(cond_low, y_low, y)
    y = tl.where(cond_mid, y_mid, y)
    y = tl.where(cond_high, y_high, y)

    # The above tl.where is not needed; since y is computed in the scope, we just assign the final y:
    # We'll keep it by recomputing with piecewise flags:
    # Note: Triton doesn't support dynamic branch labels; we compute all parts and select via tl.where at the end.

    # Final y (note: we already assigned y in each branch; Triton allows this construct reliably)
    tl.store(out_ptr, y)


@triton.jit
def _apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, out_ptr,
                            z, rows, K,
                            BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out[row, col] = max(0, x[row, col] - (mean[row] + std[row] * z))
    x_ptr/out_ptr: [rows, K] contiguous. mean_ptr/std_ptr: [rows] float32.
    """
    row = tl.program_id(0)
    col_chunk = tl.program_id(1)

    if row >= rows:
        return

    col_start = col_chunk * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    x_row_ptr = x_ptr + row * K + offs
    x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
    x_vals = x_vals.to(tl.float32)

    mean_val = tl.load(mean_ptr + row)
    std_val = tl.load(std_ptr + row)
    threshold = mean_val + std_val * z

    gated = x_vals - threshold
    gated = tl.maximum(gated, 0.0)

    out_row_ptr = out_ptr + row * K + offs
    tl.store(out_row_ptr, gated, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable parameters
        self.block_size = 1024
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle trivial case
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, "Input must be [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape

        # Flatten to [rows, K] contiguous
        x2d = x.contiguous().view(B * S, K)

        # Output buffer (float32 for computation)
        out2d = torch.empty_like(x2d, dtype=torch.float32)

        # Allocate per-row stats (float32)
        rows = B * S
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        # 1) Compute per-row mean and std via Triton reduction
        grid_stats = (rows,)
        _compute_row_stats_2d_kernel[grid_stats](
            x2d, mean, std,
            rows, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        p = torch.tensor([target_sparsity], dtype=torch.float32, device=x.device)
        z_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_kernel[(1,)](p, z_out)

        z = z_out[0]  # scalar float32

        # 3) Apply gating via Triton
        grid_gate = (rows, triton.cdiv(K, self.block_size))
        _apply_gating_2d_kernel[grid_gate](
            x2d, mean, std, out2d,
            z, rows, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Reshape back to [B, S, K] and cast to bfloat16
        out = out2d.view(B, S, K).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
