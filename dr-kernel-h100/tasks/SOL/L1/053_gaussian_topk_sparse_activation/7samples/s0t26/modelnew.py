import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_rows_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) across K for each row.
    x_ptr is a 2D view with row stride K (i.e., x_ptr[row, col] = x_ptr + row*K + col).
    """
    row = tl.program_id(0)
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over columns in chunks
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # Compute pointer for this row and chunk
        ptrs = x_ptr + row * K + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    Kf = tl.cast(K, tl.float32)
    mean = sum_val / Kf
    # population std: sqrt(E[x^2] - mean^2)
    var = sum_sq / Kf - mean * mean
    # Ensure non-negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results (same row index in mean/std vectors)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity,  # output scalar buffer
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF for target_sparsity using A&S 5.2.23.
    Writes result to z_buf[0].
    """
    p = tl.cast(target_sparsity, tl.float32)

    # Lower region
    mask_low = p < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Upper region
    mask_high = p > p_high
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    z_mid = (((((a1 * (p - 0.5) * (p - 0.5) + a2) * (p - 0.5) * (p - 0.5) + a3) * (p - 0.5) * (p - 0.5) + a4) * (p - 0.5) * (p - 0.5) + a5) * (p - 0.5) * (p - 0.5) + a6) * (p - 0.5) / \
            (((((b1 * (p - 0.5) * (p - 0.5) + b2) * (p - 0.5) * (p - 0.5) + b3) * (p - 0.5) * (p - 0.5) + b4) * (p - 0.5) * (p - 0.5) + b5) * (p - 0.5) * (p - 0.5) + 1.0)

    # Select region
    z_val = tl.where(mask_low, z_low, tl.where(mask_high, z_up, z_mid))

    # Store to output buffer (scalar)
    tl.store(z_buf, z_val)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z)), with 2D grid (rows, col_tiles).
    x_ptr, out_ptr are flattened views of [total_rows, K].
    mean_ptr, std_ptr are [total_rows].
    z_ptr is [1] scalar.
    """
    row = tl.program_id(0)
    col_tile = tl.program_id(1)
    start = col_tile * BLOCK_SIZE
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_ptr)  # scalar

    # Compute base pointers for this row
    x_row_ptr = x_ptr + row * K
    out_row_ptr = out_ptr + row * K

    # Load, compute, store
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    threshold = mean + std * z
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_row_ptr + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and std over last dim K (population std, unbiased=False).
        - Compute z = _ndtri(target_sparsity) in Triton.
        - Apply gating: relu(x - (mean + std * z)) and return in bfloat16.
        """
        # Ensure contiguous and float32 for compute
        B, S, K = x.shape
        total_rows = B * S
        x_contig = x.contiguous()

        # Allocate outputs/scratch
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

        # 1) Per-row stats: reduction across K
        # Each program handles one row; loop over K in chunks.
        BLOCK_SIZE_STATS = 4096  # works well for K up to 12288
        grid_stats = (total_rows,)
        compute_row_stats_rows_kernel[grid_stats](
            x_contig.view(total_rows, K), mean, std,
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton
        # Constants for A&S 5.2.23
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024,
            num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # read scalar without creating torch tensors on host

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K
        if K >= 8192:
            BLOCK_SIZE_GATE = 4096
            NUM_WARPS_GATE = 8
        else:
            BLOCK_SIZE_GATE = 2048
            NUM_WARPS_GATE = 4

        grid_gate = (total_rows, triton.cdiv(K, BLOCK_SIZE_GATE))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=NUM_WARPS_GATE,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior and reshape
        return out_f32.view(B, S, K).to(torch.bfloat16)