import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row (flattened [B*S]).
    x_ptr: flattened input pointer with row stride = K, accessed as row_id * K + col.
    mean_ptr, std_ptr: shape [total_rows], contiguous.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return
    acc_sum = 0.0
    acc_sum2 = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_SIZE):
        idx = k0 + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        # Load with fp32
        x = tl.load(x_ptr + row_id * K + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sum2 += tl.sum(x * x, axis=0)
    mean = acc_sum / K
    var = acc_sum2 / K - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    # Compute z = ndtri(target_sparsity) using A&S 5.2.23 approximation.
    t = target_sparsity  # scalar
    # We'll compute in fp32. Output stored as fp32.
    # Lower region
    # q = sqrt(-2 * log(p)) for p in (0, p_low)
    # Implement piecewise approximation:
    # For lower and upper tails, compute q and polynomial; mid region uses t - 0.5
    # Since this is scalar, we can directly compute using piecewise logic.
    # Triton prefers vector operations, but here it's a scalar; we'll use simple math.
    # We'll use the same piecewise logic as in the PyTorch version:
    # Lower
    # mask_low = t < p_low
    # q = sqrt(-2 * log(t))
    # poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
    # denom = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # z_lower = poly / denom
    # Upper
    # mask_high = t > p_high
    # q = sqrt(-2 * log(1 - t))
    # poly = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
    # denom = ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # Central
    # q = t - 0.5
    # r = q*q
    # poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q
    # denom = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    # z_mid = poly / denom

    # Triton scalar math: define masks and compute
    # Triton does not support dynamic branching on scalar values in the same way as Python,
    # but we can structure as follows:
    # We'll compute all pieces and select via simple comparisons (they are scalars).
    # Lower branch
    q_low = tl.sqrt(-2.0 * tl.log(t))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = poly_low / denom_low

    # Upper branch
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - t))
    poly_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    denom_up = ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    z_up = poly_up / denom_up

    # Central branch
    q_mid = t - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # Select based on t
    # If t < p_low: z = z_low
    # elif t > p_high: z = z_up
    # else: z = z_mid
    # Triton allows simple if/elif on scalars in kernels:
    if t < p_low:
        z = z_low
    elif t > p_high:
        z = z_up
    else:
        z = z_mid

    # Store to z_buf as fp32
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_buf_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z))
    x_ptr: flattened input pointer over all rows and K
    mean_ptr, std_ptr: per-row vectors of length total_rows
    z_buf_ptr: scalar pointer to z
    out_ptr: flattened output pointer
    """
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    if row_id >= total_rows:
        return
    # Load per-row stats
    mean = tl.load(mean_ptr + row_id).to(tl.float32)
    std = tl.load(std_ptr + row_id).to(tl.float32)
    z = tl.load(z_buf_ptr).to(tl.float32)
    cutoff = mean + std * z
    # Iterate over columns in this block
    for k0 in range(0, K, BLOCK_SIZE):
        idx = k0 + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        x = tl.load(x_ptr + row_id * K + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_id * K + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and std along K (feature dim).
        - Compute z = ndtri(target_sparsity) in Triton.
        - Apply gating: out = relu(x - (mean + std * z)), return bfloat16.
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return x

        # Make input contiguous and flatten rows
        B, S, K = x.shape
        total_rows = B * S
        x_contig = x.contiguous()
        # Buffers for stats (fp32)
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # 1) Compute per-row mean and std in Triton
        # Choose a reasonable BLOCK_SIZE for reduction (1024 works well broadly)
        compute_row_stats_kernel[(total_rows,)](
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=1024,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = ndtri(target_sparsity) in Triton
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        # Constants for A&S approximation
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

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            BLOCK_SIZE=1024,
            num_warps=1, num_stages=1
        )

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Use a robust, fixed BLOCK_SIZE for gating to improve consistency across K
        BLOCK_SIZE_GATE = 2048
        grid_gate = (total_rows, triton.cdiv(K, BLOCK_SIZE_GATE))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=4,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
