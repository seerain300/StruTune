import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, total_cols, K,
                              BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row in [0, total_rows).
    x is flattened as [total_rows * K] with stride per row equal to K.
    mean_ptr/std_ptr are shape [total_rows], flattened.
    """
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Base pointer for this row
    x_row_ptr = x_ptr + row_id * K
    # Masked load with other=0 to avoid invalid elements contributing to sums
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)

    # Accumulate sum and sum of squares in fp32
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    # Mean and variance (population), unbiased=False
    mean = sum_x / K
    var = sum_x2 / K - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to round-off
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity,
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF at target_sparsity into z_buf_ptr[0].
    Uses Abramowitz & Stegun 5.2.23 approximation with three regions.
    z_buf_ptr is a 1-element tensor on device; write scalar result there.
    """
    # Only one program
    # Region masks
    # Note: Triton kernels operate on vectors; we use elementwise logic
    # but here we have a single scalar target_sparsity. We'll compute for lane 0.
    lane = tl.arange(0, BLOCK_SIZE)
    # For scalar computation, construct masks as scalars by broadcasting
    # However, Triton requires tensor-like ops; implement per-lane logic.
    # Compute q for each region and select with tl.where.
    # This is a standard implementation pattern; final selection broadcasts.
    # Lower region
    mask_low = target_sparsity < p_low
    # Central region
    mask_mid = (target_sparsity >= p_low) & (target_sparsity <= p_high)
    # Upper region
    mask_high = target_sparsity > p_high

    # Common intermediates
    # Lower region computation
    q_low = tl.sqrt(-2.0 * tl.log(target_sparsity))
    poly_num_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_num_low / poly_den_low

    # Central region computation
    q_mid = target_sparsity - 0.5
    r_mid = q_mid * q_mid
    poly_num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    poly_den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_num_mid / poly_den_mid

    # Upper region computation
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
    poly_num_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    poly_den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_num_high / poly_den_high

    # Select by mask
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store to 1-element buffer
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_perrow_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                               total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Per-row kernel: for each row, compute threshold = mean + std * z,
    then out[i] = max(x[i] - threshold, 0) for i across K.
    x_ptr/out_ptr are flattened [total_rows*K], mean_ptr/std_ptr are [total_rows].
    """
    row_id = tl.program_id(0)
    # Load mean and std for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar z
    threshold = mean + std * z

    # Loop over K in chunks of BLOCK_SIZE
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K

        x_row_ptr = x_ptr + row_id * K
        x_val = tl.load(x_row_ptr + offs, mask=mask, other=0.0)

        y = x_val - threshold
        y = tl.maximum(y, 0.0)  # ReLU

        out_row_ptr = out_ptr + row_id * K
        tl.store(out_row_ptr + offs, y, mask=mask)


def _ndtri(p: float) -> float:
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
    # We compute using Triton kernels in ModelNew; this helper is only for completeness/testing.
    # For Triton path, use compute_ndtri_kernel above.
    # Placeholder: implement a correct scalar approximation here if needed.
    # We'll use a standard approximation for mid region (p in (0.02425, 0.97575)).
    # For p <= p_low, use q = sqrt(-2*log(p)); for p >= 0.97575, use q = sqrt(-2*log(1-p)).
    # To ensure correctness, prefer Triton kernel for production.
    pass


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version:
        - Per-row mean and std (reduction) in Triton.
        - Inverse normal CDF z in Triton (A&S 5.2.23).
        - Gating: out = relu(x - (mean + std * z)) in Triton per-row loop.
        - Output in bfloat16 (same as original).
        """
        # Ensure contiguous input for simple flattening
        x = x.contiguous()
        # Dimensions
        B, S, K = x.shape
        total_rows = B * S

        # 1) Compute mean and std per row in fp32
        x_flat = x.view(-1)  # length = total_rows * K
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Choose BLOCK_SIZE for reduction
        if K >= 16384:
            BLOCK_SIZE_RED = 4096
        elif K >= 8192:
            BLOCK_SIZE_RED = 4096
        else:
            BLOCK_SIZE_RED = 2048

        grid_stats = (total_rows, triton.cdiv(K, BLOCK_SIZE_RED))
        compute_row_stats_kernel[grid_stats](
            x_flat, mean, std, total_rows, K, BLOCK_SIZE=BLOCK_SIZE_RED,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (scalar)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Constants as Python floats; Triton will accept them as scalars
        a1, a2, a3, a4, a5, a6 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
        b1, b2, b3, b4, b5 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
        c1, c2, c3, c4, c5, c6 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
        d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024, num_warps=1, num_stages=1
        )

        # 3) Apply gating per row: out = relu(x - (mean + std * z))
        x_f32 = x.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Choose BLOCK_SIZE for per-row gating
        if K >= 8192:
            BLOCK_SIZE_GATE = 4096
            num_warps_gate = 8
        else:
            BLOCK_SIZE_GATE = 2048
            num_warps_gate = 4

        apply_gating_perrow_kernel[(total_rows,)](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K, BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=num_warps_gate, num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
