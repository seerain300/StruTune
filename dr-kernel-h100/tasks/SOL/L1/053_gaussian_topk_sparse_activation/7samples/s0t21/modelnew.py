import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row (flattened index pid in [0, total_rows)), compute:
      mean = (1/K) * sum(x[row, :])
      var = (1/K) * sum((x - mean)^2)
      std = sqrt(max(var, 0))
    Store results to mean_ptr[pid], std_ptr[pid].
    x_ptr is flattened with row stride = K.
    """
    pid = tl.program_id(0)
    if pid >= total_rows:
        return

    sum_x = 0.0
    sum_x2 = 0.0

    col_start = 0
    while col_start < K:
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        row_base = pid * K
        x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        col_start += BLOCK_SIZE

    mean = sum_x / K
    # population variance (unbiased=False), std is sqrt(max(var, 0))
    var = sum_x2 / K - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_ptr, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6, d1, d2, d3, d4,
                         p_low, p_high, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for p in (0, 1) using
    Abramowitz & Stegun 5.2.23 approximation. Store result to z_ptr[0].
    """
    # This kernel is a single-program launch with (1,) grid.
    # We use a compile-time tile to satisfy Triton constraints.
    # Compute z for p
    # Note: p is scalar; compute in fp32
    p_val = p
    p_low = p_low
    p_high = p_high

    # Regions
    # Lower region: p < p_low
    q = tl.sqrt(-2.0 * tl.log(p_val))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region: p_low <= p <= p_high
    u = p_val - 0.5
    r = u * u
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * u / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region: p > p_high
    q2 = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
    z_high = -(((((c1 * q2 + c2) * q2 + c3) * q2 + c4) * q2 + c5) * q2 + c6) / \
             ((((d1 * q2 + d2) * q2 + d3) * q2 + d4) * q2 + 1.0)

    # Select result based on region
    cond_low = p_val < p_low
    cond_high = p_val > p_high
    # Triton doesn't have tl.where with multi-branch; implement via masks
    z = z_low
    z = tl.where(cond_high, z_high, z)
    z = tl.where(cond_low, -z_low, z)

    # Store scalar to z_ptr[0]
    tl.store(z_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    2D tiled gating:
      row_id = program_id(0), col_tile = program_id(1)
      For each row, load mean[row] and std[row], compute threshold = mean + std * z,
      then out[row, col] = relu(x[row, col] - threshold). All in fp32.
      x_ptr, out_ptr are flattened views of [total_rows, K].
      mean_ptr, std_ptr are [total_rows].
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    col_tile = tl.program_id(1)
    col_start = col_tile * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar z
    threshold = mean + std * z

    row_base = row_id * K
    x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x - threshold, 0.0)
    tl.store(out_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # x shape: [batch_size, seq_len, intermediate_size] = [B, S, K]
        if target_sparsity == 0.0:
            # No sparsity: pass through, match dtype
            return x

        # Ensure contiguous and flatten [B, S] to rows
        B, S, K = x.shape
        total_rows = B * S
        x_contig = x.contiguous()  # keep original dtype for input
        # Allocate per-row stats in fp32
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # 1) Compute per-row mean and std in Triton (reduction)
        # Choose a reasonable block size for reduction; 2048 works well across sizes
        block_size_red = 2048
        compute_row_stats_kernel[(total_rows,)](
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=block_size_red,
            num_warps=4,
            num_stages=self.num_stages
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (scalar)
        # Create a 1-element device buffer for z
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Constants for A&S 5.2.23 approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6, d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024,  # small tile for scalar math
            num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
        x_f32 = x_contig.to(torch.float32)  # compute in fp32 for stability
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K
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