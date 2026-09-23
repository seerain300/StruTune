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

    for k in range(0, K, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sum2 += tl.sum(x * x, axis=0)

    n = K
    mean = acc_sum / n
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = acc_sum2 / n - mean * mean
    # Clamp to non-negative to avoid tiny negative due to roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(out_ptr, p, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF for p using Abramowitz & Stegun 5.2.23 approximation.
    Writes a single scalar to out_ptr[0].
    """
    # Only one program is needed; p is a scalar
    # Lower region
    mask_low = p < p_low
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                     ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    else:
        result_low = 0.0

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q = p - 0.5
        r = q * q
        result_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
                     (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    else:
        result_mid = 0.0

    # Upper region
    mask_high = p > p_high
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result_high = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                       ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    else:
        result_high = 0.0

    z = result_low + result_mid + result_high
    tl.store(out_ptr, z)


@triton.jit
def apply_gating_2d_kernel(in_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(in - (mean + std * z)), broadcast threshold per row.
    Grid: (total_rows, num_tiles), each program handles one row and one tile along K.
    """
    pid_row = tl.program_id(0)
    pid_tile = tl.program_id(1)
    row_offset = pid_row * K
    col_offset = pid_tile * BLOCK_SIZE

    offs = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load row stats and scalar z
    mean = tl.load(mean_ptr + pid_row)
    std = tl.load(std_ptr + pid_row)
    z = tl.load(z_ptr)  # single scalar

    x = tl.load(in_ptr + row_offset + offs, mask=mask, other=0.0)
    threshold = mean + std * z
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and flatten [B, S, K] -> [total_rows, K]
        total_rows = x.shape[0] * x.shape[1]
        K = x.shape[2]
        x_view = x.contiguous().view(total_rows, K)

        # Buffers for mean and std per row
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # 1) Compute row-wise mean and std using Triton reduction kernel
        # Choose BLOCK_SIZE for reduction; 1024 or 2048 are good general choices.
        BLOCK_SIZE_R = 1024
        grid_r = (total_rows,)
        compute_row_stats_kernel[grid_r](
            x_view, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_R,
            num_warps=4,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

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
            num_warps=1,
            num_stages=1
        )

        # 3) Apply gating with 2D Triton kernel
        # Compute in fp32; return in bfloat16 to match original behavior.
        in_f32 = x_view.to(torch.float32)
        out_f32 = torch.empty_like(in_f32)

        # Dynamic tuning for gating kernel based on K
        if K >= 8192:
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

        # Reshape back and cast to bfloat16
        out = out_f32.view(x.shape[0], x.shape[1], x.shape[2]).to(torch.bfloat16)
        return out