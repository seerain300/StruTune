import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) over the last dimension K.
    x_ptr: flattened input, layout is [total_rows * K], row_stride = K.
    mean_ptr, std_ptr: shape [total_rows], dtype float32.
    """
    row_id = tl.program_id(0)  # each program handles one row
    offs = tl.arange(0, BLOCK_SIZE)
    sum_row = 0.0
    sum_sq_row = 0.0
    count = 0.0  # keep count as float for averaging

    for col_start in range(0, K, BLOCK_SIZE):
        idx = row_id * K + col_start + offs
        mask = col_start + offs < K
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        sum_row += tl.sum(tl.where(mask, x_f, 0.0))
        sum_sq_row += tl.sum(tl.where(mask, x_f * x_f, 0.0))
        count += tl.sum(tl.cast(mask, tl.float32))

    mean = sum_row / count
    var = sum_sq_row / count - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high):
    """
    Inverse standard normal CDF (quantile function) using A&S 5.2.23 approximation.
    Writes result into z_buf[0] as float32.
    """
    p = target_sparsity  # scalar 0 < p < 1
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(z_buf, z)  # z_buf is a 1-element buffer


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Gating: out = relu(x - (mean + std * z)), where z = ndtri(target_sparsity).
    x_ptr: flattened input [total_rows * K], float32
    mean_ptr, std_ptr: [total_rows], float32
    z_ptr: 1-element buffer, float32
    out_ptr: flattened output [total_rows * K], float32
    """
    row_id = tl.program_id(0)  # each program handles one row
    z = tl.load(z_ptr)  # scalar

    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        idx = row_id * K + offs

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + row_id)
        std = tl.load(std_ptr + row_id)

        thr = mean + std * z
        gated = x - thr
        gated = tl.maximum(gated, 0.0)  # ReLU
        tl.store(out_ptr + idx, gated, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, "Input must be of shape [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape
        total_rows = B * S

        # Flatten to [rows, K], make contiguous for linear indexing
        x_flat = x.contiguous().view(total_rows, K)

        # Allocate outputs and stats in fp32 for numerical stability
        mean_buf = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std_buf = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        out_f32 = torch.empty((total_rows, K), device=x.device, dtype=torch.float32)

        # 1) Compute per-row mean and std in Triton
        # Choose BLOCK_SIZE based on K
        if K >= 8192:
            block_size_stats = 4096
            num_warps_stats = 8
            num_stages_stats = 2
        elif K >= 4096:
            block_size_stats = 2048
            num_warps_stats = 4
            num_stages_stats = 2
        else:
            block_size_stats = 1024
            num_warps_stats = 4
            num_stages_stats = 1

        compute_row_stats_kernel[(total_rows,)](
            x_flat.view(-1), mean_buf, std_buf, total_rows, K,
            BLOCK_SIZE=block_size_stats,
            num_warps=num_warps_stats,
            num_stages=num_stages_stats
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton, write to 1-element buffer
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        # constants for A&S 5.2.23
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity), a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high,
            num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
        # Choose BLOCK_SIZE based on K for gating
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
            x_flat.view(-1), mean_buf, std_buf, z_buf, out_f32.view(-1), total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=num_stages_gate
        )

        # Reshape and cast back to bfloat16 to match original behavior
        out = out_f32.view(B, S, K).to(torch.bfloat16)
        return out