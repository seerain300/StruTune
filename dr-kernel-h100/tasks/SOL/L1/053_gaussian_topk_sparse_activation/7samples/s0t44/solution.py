import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row (flattened index in [0, total_rows)), compute:
      mean = sum(x[0:K]) / K
      std  = sqrt(sum((x[0:K] - mean)^2) / K)  (population std, unbiased=False)
    x_ptr: pointer to flattened input of length total_rows * K
    mean_ptr, std_ptr: per-row outputs of length total_rows
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return
    row_base = row_id * K

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over K in tiles
    for offs in range(0, K, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    # Clamp small negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4, p_low):
    # Single-program computation of z = ndtri(target_sparsity) using A&S 5.2.23.
    p = target_sparsity
    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / den_low

    # Central region
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    mask_high = p > (1.0 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / den_high

    # Select appropriate branch
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store scalar
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row i in [0, total_rows), for each column j in [0, K):
      t = mean[i] + std[i] * z
      out[i, j] = relu(x[i, j] - t)
    x_ptr: flattened input pointer, length total_rows * K
    mean_ptr, std_ptr: per-row values, length total_rows
    z_ptr: 1-element buffer containing scalar z
    out_ptr: flattened output pointer, length total_rows * K
    """
    row_id = tl.program_id(0)  # each program handles one row
    col_block = tl.program_id(1)  # each program also handles one column tile
    if row_id >= total_rows:
        return

    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load row scalars
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar

    t = mean + std * z

    # Load input row slice
    row_base = row_id * K
    x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Gating: relu(x - t)
    y = tl.maximum(x - t, 0.0)

    # Store
    tl.store(out_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Validate inputs
        if target_sparsity < 0.0 or target_sparsity > 1.0:
            raise ValueError("target_sparsity must be in [0, 1]")
        if target_sparsity == 0.0:
            # No sparsity; return input as-is (bfloat16)
            return x

        # Ensure 3D [B, S, K]
        if x.dim() != 3:
            raise ValueError("Input must be a 3D tensor [batch_size, seq_len, intermediate_size]")
        B, S, K = x.shape
        total_rows = B * S

        # Ensure contiguous in last dimension for efficient row-wise reduction
        x_contig = x.contiguous()

        # 1) Compute per-row mean and std in fp32
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Choose reduction tile; 2048 is a good default
        BLOCK_SIZE_STATS = 2048
        compute_row_stats_kernel[(total_rows,)](  # one program per row
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = ndtri(target_sparsity) in Triton (scalar)
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

        # Constants for A&S 5.2.23
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
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
            num_warps=1, num_stages=1
        )

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x_contig.to(torch.float32)  # compute in fp32
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K for better throughput
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
        else:
            block_size_gate = 2048
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
