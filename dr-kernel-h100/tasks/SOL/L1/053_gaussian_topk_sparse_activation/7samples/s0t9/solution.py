import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_2d_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, stride_row, stride_col, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (per [b, s]) mean and std along the last dimension (K).
    x_ptr points to a [total_rows, K] view with given strides (stride_row for rows, stride_col for columns).
    mean_ptr, std_ptr are vectors of length total_rows (we'll view as [B, S, 1] in host).
    """
    row = tl.program_id(0)  # 0..total_rows-1
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over K in chunks of BLOCK_SIZE
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # Compute linear offsets: row * stride_row + cols * stride_col
        offsets = row * stride_row + cols * stride_col
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
        sum_sq += tl.sum(vals * vals)

    n = K
    mean = sum_val / n
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / n - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to round-off
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high, BLOCK_SIZE: tl.constexpr):
    # Compute inverse standard normal CDF z for given sparsity using A&S 5.2.23 approximation.
    # We run a single program and write to z_buf_ptr[0]
    # Note: target_sparsity is a scalar argument; no torch usage in host.

    # Lower region
    if target_sparsity < p_low:
        q = tl.sqrt(-2.0 * tl.log(target_sparsity))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # Central region
        x = target_sparsity - 0.5
        r = x * x
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly * x / denom
        # Upper region
        if target_sparsity > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
            z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K,
                           BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z)), elementwise over [total_rows, K].
    mean_ptr, std_ptr are vectors of length total_rows. z_ptr is a 1-element tensor holding scalar z.
    """
    row = tl.program_id(0)  # 0..total_rows-1
    col_tile = tl.program_id(1)  # 0..ceil_div(K, BLOCK_SIZE)-1

    off = col_tile * BLOCK_SIZE
    cols = off + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load per-row mean and std
    mean_row = tl.load(mean_ptr + row)
    std_row = tl.load(std_ptr + row)
    z_val = tl.load(z_ptr)  # scalar

    threshold = mean_row + std_row * z_val
    # Build per-row base offsets
    # We assume x/out are flattened row-major with stride_row = K, stride_col = 1
    base = row * K
    offsets = base + cols
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = x_vals - threshold
    # ReLU
    y_vals = tl.maximum(y_vals, 0.0)
    tl.store(out_ptr + offsets, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        # Defaults; will be tuned dynamically in forward based on K
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the original run function.
        - Computes per-row mean and std in Triton.
        - Computes z = _ndtri(target_sparsity) in Triton.
        - Applies gating in Triton and returns bfloat16 output.
        """
        assert x.dim() == 3, "Input must be 3D [B, S, K]"
        B, S, K = x.shape
        total_rows = B * S

        # Ensure contiguous memory for Triton kernels
        x_contig = x.contiguous()

        # 1) Compute per-row mean and std in Triton
        mean = torch.empty((total_rows,), device=x.device, dtype=torch.float32)
        std = torch.empty((total_rows,), device=x.device, dtype=torch.float32)

        # We'll treat x as a 2D [total_rows, K] view with strides (stride_row=K, stride_col=1)
        stride_row = K
        stride_col = 1

        # Choose block size based on K (dynamic tuning)
        if K >= 16384:
            block_size_stats = 1024  # reduction block size
            block_size_gate = 4096
            num_warps_stats = 4
            num_warps_gate = 8
        elif K >= 8192:
            block_size_stats = 1024
            block_size_gate = 4096
            num_warps_stats = 4
            num_warps_gate = 4
        elif K >= 4096:
            block_size_stats = 1024
            block_size_gate = 2048
            num_warps_stats = 4
            num_warps_gate = 4
        else:
            block_size_stats = 1024
            block_size_gate = 1024
            num_warps_stats = 4
            num_warps_gate = 4

        compute_row_stats_2d_kernel[(total_rows,)](
            x_contig, mean, std,
            total_rows, K, stride_row, stride_col,
            BLOCK_SIZE=block_size_stats,
            num_warps=num_warps_stats,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (single program)
        z_buf = torch.empty((1,), device=x.device, dtype=torch.float32)

        # Constants for A&S 5.2.23 approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            BLOCK_SIZE=1024,
            num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
