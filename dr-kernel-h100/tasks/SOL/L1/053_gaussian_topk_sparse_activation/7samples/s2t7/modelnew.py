import torch
import triton
import triton.language as tl


# Triton reduction kernel: per-row mean and std across last dim
# x_flat: [rows, N], mean[rows], std[rows]
@triton.jit
def reduce_mean_std(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Offsets for a chunk
    offs = tl.arange(0, BLOCK_SIZE)
    # Accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    # Loop over columns
    for col in range(0, N, BLOCK_SIZE):
        idx = row_id * N + col + offs
        mask = (col + offs) < N
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    # Compute mean and std
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    std = tl.sqrt(var)
    # Write as 1-element per row
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


# Triton scalar kernel: compute inv_norm_cdf(target_sparsity) using A&S approximation
# Writes to out_ptr[0]
@triton.jit
def compute_inv_ndtri(out_ptr, target_sparsity):
    # Constants for approximation
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

    p = target_sparsity

    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_low = poly_low / ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    inv_low = -poly_low

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid_num = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_mid_den = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    inv_mid = q_mid * poly_mid_num / poly_mid_den

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    poly_high = poly_high / ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    inv_high = poly_high

    inv = tl.where(mask_low, inv_low, 0.0)
    inv = tl.where(mask_mid, inv_mid, inv)
    inv = tl.where(mask_high, inv_high, inv)

    tl.store(out_ptr, inv)


# Triton gating kernel: per-row thresholding
# x_flat: [rows, N], mean[rows], std[rows], inv_scale (scalar), out_flat: [rows, N]
@triton.jit
def gate_rows(x_ptr, mean_ptr, std_ptr, inv_scale, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    for col in range(0, N, BLOCK_SIZE):
        idx = row_id * N + col + offs
        mask = (col + offs) < N
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + row_id)
        std = tl.load(std_ptr + row_id)
        threshold = mean + std * inv_scale
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure contiguous and compute in float32
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape
        rows = B * S

        # Flatten to [rows, N]
        x_flat = x_f32.view(rows, N)
        # Output buffer (float32 for computation)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # 1) Reduce to per-row mean and std
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std[grid](x_flat, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri[(1,)](inv_cdf_buf, target_sparsity)

        # Scale factor for gating: inv_cdf * target_sparsity
        inv_scale = float(inv_cdf_buf.item()) * float(target_sparsity)

        # 3) Gate rows: y = max(0, x - (mean + std * inv_scale))
        gate_rows[grid](x_flat, mean, std, inv_scale, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out