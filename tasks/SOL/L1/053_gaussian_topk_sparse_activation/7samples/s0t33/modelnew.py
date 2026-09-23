import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std (population std, unbiased=False) along last dim K.
    x_ptr points to flattened [total_rows*K] row-major data with stride K between rows.
    Writes mean and std as [total_rows] float32.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    # Accumulators in int64 for numeric stability and correct dtype with K
    sum_x = tl.zeros((), dtype=tl.int64)
    sum_x2 = tl.zeros((), dtype=tl.int64)

    start = 0
    while start < K:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Address for this row: base = row_id * K, then + offs
        x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
        x = x.to(tl.int64)  # accumulate in int64
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    # Convert to float32 for mean/std
    n = tl.full((), K, tl.float32)
    mean = sum_x.to(tl.float32) / n
    var = sum_x2.to(tl.float32) / n - mean * mean
    var = tl.maximum(var, 0.0)  # clamp to avoid tiny negative due to fp errors
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, p, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for p in (0,1).
    Uses Abramowitz and Stegun 5.2.23 approximation.
    Writes a single float to z_buf_ptr[0].
    """
    # Single program instance; masked loads/stores not needed here
    # Compute z = inverse CDF for given p using A&S 5.2.23 approximation
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    poly1 = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    poly2 = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    z_lower = poly1 / poly2

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    poly3 = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly4 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_center = poly3 * q2 / poly4

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly5 = (((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6)
    poly6 = ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)
    z_upper = -poly5 / poly6

    # Combine with piecewise masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    # Select result based on masks
    z = tl.where(mask_low, z_lower, 0.0)
    z = tl.where(mask_mid, z_center, z)
    z = tl.where((p > (1.0 - p_low)), z_upper, z)

    # Store result
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out[row, col] = relu(x[row, col] - (mean[row] + std[row] * z))
    x_ptr, out_ptr are flattened [total_rows*K].
    mean_ptr, std_ptr are [total_rows].
    z_ptr is [1] float32.
    """
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    if row_id >= total_rows:
        return

    start = col_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar

    x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    threshold = mean + std * z
    out = tl.maximum(x - threshold, 0.0)

    tl.store(out_ptr + row_id * K + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    # Constants for Abramowitz & Stegun 5.2.23 approximation (5.2.23)
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

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation:
        - Compute per-row mean and std along last dim (intermediate_size).
        - Compute z = _ndtri(target_sparsity) in Triton.
        - Apply gating: out = relu(x - (mean + std * z)), return bfloat16.
        """
        assert x.dim() == 3, "Input must be [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape
        total_rows = B * S

        # Make input contiguous and compute in fp32 for stability
        x_contig = x.contiguous()
        x_fp32 = x_contig.to(torch.float32)

        # 1) Compute mean and std per row (population std, unbiased=False)
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Reduction kernel launch
        BLOCK_SIZE = 1024  # 1024 works well across sizes; adjust if needed
        grid = (total_rows,)
        compute_row_stats_kernel[grid](
            x_fp32.view(-1), mean, std, total_rows, K, BLOCK_SIZE,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            self.a1, self.a2, self.a3, self.a4, self.a5, self.a6,
            self.b1, self.b2, self.b3, self.b4, self.b5,
            self.c1, self.c2, self.c3, self.c4, self.c5, self.c6,
            self.d1, self.d2, self.d3, self.d4,
            self.p_low, 1.0 - self.p_low,
            BLOCK_SIZE=1024, num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
        out_f32 = torch.empty_like(x_fp32)

        # Dynamic tuning for gating kernel based on K for robustness
        if K >= 16384:
            block_size_gate = 4096
            num_warps_gate = 4
        elif K >= 8192:
            block_size_gate = 2048
            num_warps_gate = 4
        else:
            block_size_gate = 1024
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_fp32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)