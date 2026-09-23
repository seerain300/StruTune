import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row (flattened [B*S]).
    x_ptr: flattened input pointer, stride per row is K.
    mean_ptr, std_ptr: shape [total_rows] (we pass [B*S, 1] and take view of length total_rows).
    """
    row_id = tl.program_id(0)
    acc1 = 0.0  # sum
    acc2 = 0.0  # sum of squares

    # Loop over columns in tiles of BLOCK_SIZE
    for start in range(0, K, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        # Row base offset: row_id * K
        offs = row_id * K + idx
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc1 += tl.sum(vals, axis=0)
        acc2 += tl.sum(vals * vals, axis=0)

    mean = acc1 / K
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = acc2 / K - mean * mean
    # Ensure non-negative to avoid tiny negative due to FP rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results at [row_id]
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for target_sparsity using A&S 5.2.23 approximation.
    Writes the result into z_buf_ptr[0] as a scalar.
    """
    t = target_sparsity  # scalar
    # Piecewise approximation
    # Lower region
    mask_low = t < p_low
    q_low = tl.sqrt(-2.0 * tl.log(t))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    mask_mid = (t >= p_low) & (t <= p_high)
    q_mid = t - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / den_mid

    # Upper region
    mask_high = t > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - t))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select piecewise result
    cond = mask_low | mask_mid | mask_high
    # We need to combine piecewise results; Triton supports tl.where
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store scalar to z_buf_ptr[0]
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_buf_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out = relu(x - (mean + std * z)), where mean, std are per row.
    x_ptr, out_ptr: flattened pointers of length total_rows * K.
    mean_ptr, std_ptr: flattened pointers of length total_rows.
    z_buf_ptr: scalar pointer (1 element) containing z = _ndtri(target_sparsity).
    """
    row_id = tl.program_id(0)
    col_tile = tl.program_id(1)
    start = col_tile * BLOCK_SIZE
    idx = start + tl.arange(0, BLOCK_SIZE)
    mask = idx < K

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    # Load z scalar
    z = tl.load(z_buf_ptr)

    # Compute threshold per element: mean + std * z
    threshold = mean + std * z
    # Load x row-wise
    offs = row_id * K + idx
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x - threshold, 0.0)  # ReLU
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float = 0.0) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        1) Compute per-row mean and std (population std).
        2) Compute z = inverse standard normal CDF of target_sparsity using A&S approximation (Triton).
        3) Apply gating: out = relu(x - (mean + std * z)) in Triton.
        Returns tensor of shape [B, S, K] with dtype bfloat16.
        """
        if target_sparsity == 0.0:
            return x

        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, "Input must be a 3D tensor [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape
        total_rows = B * S

        # Make contiguous and compute in fp32 for stability
        x_contig = x.contiguous()
        x_f32 = x_contig.to(torch.float32)

        # Allocate output
        out_f32 = torch.empty_like(x_f32)

        # 1) Compute per-row mean and std in Triton
        mean_buf = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std_buf = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Choose reduction BLOCK_SIZE based on K
        if K >= 8192:
            block_size_reduce = 4096
            num_warps_reduce = 8
            num_stages_reduce = 2
        else:
            block_size_reduce = 2048
            num_warps_reduce = 4
            num_stages_reduce = 2

        grid_reduce = (total_rows,)
        compute_row_stats_kernel[grid_reduce](
            x_f32.view(-1), mean_buf, std_buf, total_rows, K,
            BLOCK_SIZE=block_size_reduce,
            num_warps=num_warps_reduce,
            num_stages=num_stages_reduce
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (write to 1-element buffer)
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

        # Constants for A&S approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        # Launch compute_ndtri_kernel; it writes scalar to z_buf[0]
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, 1.0 - p_low,
            num_warps=1, num_stages=1
        )

        # Read z as float (no torch ops, minimal host math)
        z = float(z_buf.item())

        # 3) Apply gating with 2D Triton kernel
        # Choose BLOCK_SIZE for gating based on K
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
            num_stages_gate = 2
        else:
            block_size_gate = 2048
            num_warps_gate = 4
            num_stages_gate = 2

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean_buf, std_buf, z_buf, out_f32.view(-1), total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=num_stages_gate
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
