import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row index pid in [0, total_rows), compute mean and std over K elements.
    x_ptr points to a flattened [total_rows, K] region; stride per row is K.
    """
    pid = tl.program_id(0)
    row_offset = pid * K
    acc_sum = 0.0
    acc_sum2 = 0.0

    # Process the row in tiles of BLOCK_SIZE
    for k in range(0, K, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sum2 += tl.sum(x * x, axis=0)

    n = tl.float32(K)
    mean = acc_sum / n
    var = acc_sum2 / n - mean * mean
    # Clamp small negative due to floating-point error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high, BLOCK_SIZE: tl.constexpr):
    # Single-program approximation computation and store into z_buf_ptr[0]
    # Use A&S 5.2.23 approximation for the inverse standard normal CDF
    t = target_sparsity  # scalar in (0, 1)

    # Lower region
    mask_low = t < p_low
    q_low = tl.sqrt(-2.0 * tl.log(t))
    s_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = -s_low

    # Central region
    mask_mid = (t >= p_low) & (t <= p_high)
    q_mid = t - 0.5
    r = q_mid * q_mid
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = poly * q_mid / poly2

    # Upper region
    mask_high = t > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - t))
    s_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = s_high

    # Select
    z = tl.where(mask_low, z_low, tl.where(mask_mid, z_mid, z_high))

    # Store scalar z
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row pid in [0, total_rows), for each column tile:
    out[row, col] = relu(x[row, col] - (mean[row] + std[row] * z))
    x_ptr, out_ptr are flattened over [total_rows, K].
    mean_ptr, std_ptr are length total_rows.
    z_ptr points to a single scalar z.
    """
    pid = tl.program_id(0)
    # Determine tile start and compute scalar threshold for this row
    row_offset = pid * K
    z = tl.load(z_ptr)  # scalar
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * z

    # Tile over K
    for k in range(0, K, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        # gating: relu(x - threshold) with broadcast of threshold
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation of:
          - Compute per-row mean and std along last dim K (population std, unbiased=False).
          - Compute z = inverse standard normal CDF of target_sparsity using A&S 5.2.23.
          - Apply gating: out = relu(x - (mean + std * z)), return in bfloat16.
        """
        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, f"Expected 3D tensor, got shape {x.shape}"
        B, S, K = x.shape
        total_rows = B * S

        # Make contiguous and flatten for kernels
        x_view = x.contiguous()

        # Allocate mean and std buffers (fp32)
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # 1) Compute per-row mean and std
        # Dynamic BLOCK_SIZE for reduction
        if K >= 16384:
            block_size_red = 2048  # balance occupancy and register pressure
            num_warps_red = 4
        else:
            block_size_red = 1024
            num_warps_red = 4

        compute_row_stats_kernel[(total_rows,)](
            x_view.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=block_size_red,
            num_warps=num_warps_red,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton, store into 1-element buffer
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

        # A&S constants for ndtri
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
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
            num_warps=1,
            num_stages=1
        )

        z = z_buf.item()  # read scalar without torch tensor creation on host

        # 3) Apply gating with 2D Triton kernel, dynamic tiling based on K
        in_f32 = x_view.to(torch.float32)  # compute in fp32
        out_f32 = torch.empty_like(in_f32)

        if K >= 16384:
            block_size_gate = 4096
            num_warps_gate = 8
        else:
            block_size_gate = 2048
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            in_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.view(B, S, K).to(torch.bfloat16)