import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row (flattened index pid in [0, total_rows)), compute:
      mean = (1/K) * sum(x[row, :])
      var = (1/K) * sum((x - mean)^2)
      std = sqrt(max(var, 0))  # population std, unbiased=False
    x_ptr is flattened with row stride = K.
    """
    pid = tl.program_id(0)
    if pid >= total_rows:
        return

    # Accumulate sum and sum of squares
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over columns in tiles using a while loop (runtime K)
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

    Kf = tl.float32(K)
    mean = sum_x / Kf
    var = sum_x2 / Kf - mean * mean
    # numerical guard
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity, p_low, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF using Abramowitz & Stegun 5.2.23 approximation.
    z_buf[0] = inv_cdf(target_sparsity). Assumes p_low = 0.02425, p_high = 1 - p_low.
    """
    # constants
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # pass only required args: target_sparsity (p) and p_low
    # lower region
    if target_sparsity < p_low:
        q = tl.sqrt(-2.0 * tl.log(target_sparsity))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / den
    else:
        q = target_sparsity - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly / den
    # store to z_buf[0] (scalar)
    tl.store(z_buf, z.to(tl.float32))


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                            total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over flattened [total_rows * K]:
      out[i] = relu(x[i] - (mean[i//K] + std[i//K] * z[0]))
    x_ptr points to input (float32), out_ptr points to output (float32).
    mean_ptr and std_ptr are length total_rows.
    z_ptr is a 1-element tensor containing z scalar.
    """
    pid_row = tl.program_id(0)  # row index
    pid_col = tl.program_id(1)  # tile index along columns
    if pid_row >= total_rows:
        return

    col_start = pid_col * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    row_base = pid_row * K
    x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # load per-row mean and std
    mean = tl.load(mean_ptr + pid_row)
    std = tl.load(std_ptr + pid_row)

    # load scalar z
    z_scalar = tl.load(z_ptr)

    threshold = mean + std * z_scalar
    y = x - threshold
    # relu
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # No torch math on host; all computation in Triton kernels.

        # Ensure input is contiguous and work on its flattened [B*S, K]
        B, S, K = x.shape
        total_rows = B * S
        x_contig = x.contiguous()

        # 1) Compute per-row mean and std in Triton
        x_f32 = x_contig.to(torch.float32)
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Dynamic BLOCK_SIZE for reduction
        if K >= 16384:
            block_size_reduce = 4096
            num_warps_reduce = 8
        elif K >= 8192:
            block_size_reduce = 4096
            num_warps_reduce = 8
        else:
            block_size_reduce = 2048
            num_warps_reduce = 4

        compute_row_stats_kernel[(total_rows,)](
            x_f32.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=block_size_reduce,
            num_warps=num_warps_reduce, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton and store to 1-element buffer
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        # p_low constant as per A&S 5.2.23
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity), p_low,
            BLOCK_SIZE=1024,  # small tile for scalar math
            num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
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


def run(*args):
    return ModelNew()(*args)
