import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std across the last dim K for each row.
    x_ptr is a flattened view of shape [total_rows, K], row stride = K.
    """
    pid = tl.program_id(axis=0)  # one program per row
    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over columns in tiles of BLOCK_SIZE
    # We unroll with a static range to keep compile-time friendly for large K
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Row index in flattened array: row * K + col
        x_row = x_ptr + pid * K
        vals = tl.load(x_row + offs, mask=mask, other=0.0)
        # Reduce tile to scalars
        sum_val += tl.sum(vals.to(tl.float32))
        sum_sq += tl.sum((vals.to(tl.float32) * vals.to(tl.float32)))

    n = K  # population std, unbiased=False
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # Ensure non-negative variance for numerical safety
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results
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
    Compute inverse standard normal CDF for a single p in (0,1) using A&S 5.2.23.
    Write result to out_ptr[0].
    """
    # Single program, one scalar result
    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    r_low = q_low * q_low
    poly_low = (((((c1 * r_low + c2) * r_low + c3) * r_low + c4) * r_low + c5) * r_low + c6)
    poly_low2 = (((((d1 * r_low + d2) * r_low + d3) * r_low + d4) * r_low + 1.0))
    y_low = poly_low / poly_low2

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_mid2 = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    y_mid = poly_mid / poly_mid2

    # Upper region
    mask_high = p > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    r_high = q_high * q_high
    poly_high = (((((c1 * r_high + c2) * r_high + c3) * r_high + c4) * r_high + c5) * r_high + c6)
    poly_high2 = (((((d1 * r_high + d2) * r_high + d3) * r_high + d4) * r_high + 1.0))
    y_high = -poly_high / poly_high2

    # Select based on region; use 0.0 for inactive masks
    y = tl.where(mask_low, y_low, 0.0) + tl.where(mask_mid, y_mid, 0.0) + tl.where(mask_high, y_high, 0.0)
    tl.store(out_ptr, y)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over [total_rows, K]:
      out[i, j] = max(x[i, j] - (mean[i] + std[i] * z), 0)
    x_ptr, out_ptr are flattened [total_rows*K], mean_ptr, std_ptr are [total_rows], z_ptr is scalar.
    """
    row_id = tl.program_id(axis=0)
    col_tile = tl.program_id(axis=1)

    col_start = col_tile * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Compute cutoff for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z

    x_row = x_ptr + row_id * K
    out_row = out_ptr + row_id * K

    x = tl.load(x_row + offs, mask=mask, other=0.0)
    res = x - cutoff
    res = tl.maximum(res, 0.0)  # ReLU
    tl.store(out_row + offs, res, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation of Gaussian-based top-k sparse activation:
        1) Compute per-row mean and std over K.
        2) Compute z = inverse normal CDF of target_sparsity.
        3) Apply gating: out = relu(x - (mean + std * z)), return bfloat16.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.ndim == 3, "Expected input of shape [batch_size, seq_len, intermediate_size]."
        B, S, K = x.shape
        total_rows = B * S

        # Ensure contiguous
        x_contig = x.contiguous()

        # Buffers for mean and std (float32)
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Constants for inverse normal CDF
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

        # 1) Compute per-row mean and std with Triton
        # Choose a BLOCK_SIZE; 2048 works well across K sizes
        BLOCK_SIZE_STATS = 2048
        compute_row_stats_kernel[(total_rows,)](x_contig.view(-1), mean, std, total_rows, K,
                                                BLOCK_SIZE=BLOCK_SIZE_STATS,
                                                num_warps=4, num_stages=2)

        # 2) Compute z = _ndtri(target_sparsity) with Triton (single program)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_ndtri_kernel[(1,)](z_buf, float(target_sparsity),
                                   a1, a2, a3, a4, a5, a6,
                                   b1, b2, b3, b4, b5,
                                   c1, c2, c3, c4, c5, c6,
                                   d1, d2, d3, d4,
                                   p_low, p_high,
                                   BLOCK_SIZE=1024,
                                   num_warps=1, num_stages=1)
        z = float(z_buf.item())  # read scalar without torch.tensor in host

        # 3) Apply gating with Triton (2D, tuned dynamically based on K)
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
        else:
            block_size_gate = 2048
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, torch.tensor(z, dtype=torch.float32, device=x.device), out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
