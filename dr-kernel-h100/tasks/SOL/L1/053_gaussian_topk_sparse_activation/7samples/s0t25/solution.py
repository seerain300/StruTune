import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_rows_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one row (b, s). Reduce across K in chunks of BLOCK_SIZE to compute:
      mean = sum(x) / K
      std  = sqrt(sum(x^2)/K - mean^2)  (population std, unbiased=False)
    x_ptr: flattened pointer to input tensor of shape [total_rows, K]
    mean_ptr, std_ptr: output arrays of shape [total_rows], contiguous.
    """
    row_id = tl.program_id(0)
    # Early exit if row_id >= total_rows (usually not necessary if grid matches, but safe)
    if row_id >= total_rows:
        return

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate across K in tiles of BLOCK_SIZE
    for offs in range(0, K, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        # Pointer to this row at offsets idx
        row_base = row_id * K
        x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0)
        # Accumulate sums
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    # population std, unbiased=False
    std = tl.sqrt(sum_sq / K - mean * mean)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity,
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a single scalar target_sparsity using
    Abramowitz & Stegun 5.2.23 approximation. Store the result in z_buf[0].
    """
    # We only launch 1 program; single scalar math. BLOCK_SIZE not used for scalars.
    p = target_sparsity
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den_low = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    z_low = poly_low / den_low

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / den_high

    # Select region based on p
    select_low = p < p_low
    select_high = p > p_high

    # Use mid as default, add low/high corrections where appropriate
    z = z_mid
    z = tl.where(select_low, z_low, z)
    z = tl.where(select_high, z_high, z)

    tl.store(z_buf, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    2D tiled kernel over rows and columns to apply gating:
      out[row, col] = max(0, x[row, col] - (mean[row] + std[row] * z))
    x_ptr, out_ptr: flattened pointers of shape [total_rows * K]
    mean_ptr, std_ptr: shape [total_rows]
    z_ptr: shape [1] (scalar)
    """
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    if row_id >= total_rows:
        return

    col_start = col_block * BLOCK_SIZE
    idx = col_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < K

    row_base = row_id * K
    x = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0)

    # Load row stats and scalar z
    mean_row = tl.load(mean_ptr + row_id)
    std_row = tl.load(std_ptr + row_id)
    z_scalar = tl.load(z_ptr)  # scalar

    threshold = mean_row + std_row * z_scalar
    out = tl.maximum(x - threshold, 0.0)

    tl.store(out_ptr + row_base + idx, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure 3D input: [batch_size, seq_len, intermediate_size]
        assert x.dim() == 3, "Input must be a 3D tensor [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape
        total_rows = B * S

        # Make contiguous 2D view for row-wise kernels
        x_2d = x.contiguous().view(total_rows, K)

        # Allocate outputs and stats in fp32
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        out_f32 = torch.empty_like(x_2d, dtype=torch.float32, device=x.device)

        # Buffer for z (scalar)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # 1) Compute per-row mean and std via Triton kernel
        # Choose BLOCK_SIZE for reduction. 1024 is a good default; loop covers K regardless of size.
        BLOCK_SIZE_STATS = 1024
        compute_row_stats_rows_kernel[(total_rows,)](
            x_2d, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        # Constants for A&S 5.2.23 approximation
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

        compute_ndtri_kernel[(1,)](
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024,  # scalar kernel, BLOCK_SIZE not used here
            num_warps=1, num_stages=1
        )

        # 3) Apply gating with 2D Triton kernel
        # Dynamic tuning based on K
        if K >= 8192:
            BLOCK_SIZE_GATE = 4096
            NUM_WARPS_GATE = 8
        else:
            BLOCK_SIZE_GATE = 2048
            NUM_WARPS_GATE = 4

        grid_gate = (total_rows, triton.cdiv(K, BLOCK_SIZE_GATE))
        apply_gating_2d_kernel[grid_gate](
            x_2d.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=NUM_WARPS_GATE,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.view(B, S, K).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
