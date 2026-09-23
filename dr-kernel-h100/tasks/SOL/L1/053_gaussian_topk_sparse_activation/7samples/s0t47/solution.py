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
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = acc_sum2 / n - mean * mean
    std = tl.sqrt(var)
    # Store results for this row
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Compute inverse standard normal CDF for a given target_sparsity (scalar in [0,1])
    using Abramowitz & Stegun 5.2.23 approximation and store into z_buf[0].
    """
    # Since grid is (1,), we can compute and store to z_buf[0]
    t = target_sparsity
    # Lower region approximation
    q = tl.sqrt(-2.0 * tl.log(t))
    poly5 = c1 * q + c2
    poly4 = poly5 * q + c3
    poly3 = poly4 * q + c4
    poly2 = poly3 * q + c5
    poly1 = poly2 * q + c6
    denom5 = d1 * q + d2
    denom4 = denom5 * q + d3
    denom3 = denom4 * q + d4
    denom2 = denom3 * q + 1.0
    z_low = poly1 / denom2

    # Central region approximation
    q2 = t - 0.5
    r2 = q2 * q2
    poly5c = a1 * r2 + a2
    poly4c = poly5c * r2 + a3
    poly3c = poly4c * r2 + a4
    poly2c = poly3c * r2 + a5
    poly1c = poly2c * r2 + a6
    denom5c = b1 * r2 + b2
    denom4c = denom5c * r2 + b3
    denom3c = denom4c * r2 + b4
    denom2c = denom3c * r2 + b5
    z_mid = poly1c / denom2c

    # Upper region approximation
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - t))
    poly5u = c1 * q3 + c2
    poly4u = poly5u * q3 + c3
    poly3u = poly4u * q3 + c4
    poly2u = poly3u * q3 + c5
    poly1u = poly2u * q3 + c6
    denom5u = d1 * q3 + d2
    denom4u = denom5u * q3 + d3
    denom3u = denom4u * q3 + d4
    denom2u = denom3u * q3 + 1.0
    z_up = -poly1u / denom2u

    # Select region based on t
    # We implement piecewise selection with masks
    sel_low = t < p_low
    sel_high = t > p_high
    z = tl.zeros((), dtype=tl.float32)
    # If lower: z_low; elif upper: z_up; else: z_mid
    z = tl.where(sel_low, z_low, z)
    z = tl.where(sel_high, z_up, z)
    z = tl.where((~sel_low) & (~sel_high), z_mid, z)

    # Store single scalar
    tl.store(z_buf, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K,
                           BLOCK_SIZE: tl.constexpr):
    """
    2D tiling over rows and column tiles:
    pid_row in [0, total_rows), pid_col in [0, ceil_div(K, BLOCK_SIZE)).
    For each tile, compute threshold = mean[pid_row] + std[pid_row] * z,
    then out[r, c] = relu(x[r, c] - threshold).
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    # Compute column indices for this tile
    c = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = c < K
    # Load mean and std for this row
    mean = tl.load(mean_ptr + pid_row)
    std = tl.load(std_ptr + pid_row)
    z = tl.load(z_ptr)  # scalar
    threshold = mean + std * z
    # Row base offset in flattened [total_rows, K] is pid_row * K
    base = pid_row * K
    x = tl.load(x_ptr + base + c, mask=mask, other=0.0)
    y = tl.maximum(x - threshold, 0.0)  # relu
    # Store to out at [pid_row, c]
    tl.store(out_ptr + pid_row * K + c, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: float in [0, 1], target sparsity level
        Returns: sparsified tensor with same shape as input, dtype bfloat16.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguity and keep original shape metadata
        B, S, K = x.shape
        x_contig = x.contiguous()
        total_rows = B * S

        # 1) Compute per-row mean and std in float32 using Triton
        x_flat = x_contig.view(-1)  # [total_rows * K]
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        # Use a moderate block size for reduction; 1024 works well across K
        compute_row_stats_kernel[(total_rows,)](
            x_flat, mean, std, total_rows, K,
            BLOCK_SIZE=1024,
            num_warps=4,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        # Prepare constants
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

        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            num_warps=1,
            num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid and direct 3D writes
        in_f32 = x_contig.to(torch.float32)  # compute in fp32
        # Allocate output as 3D and write directly via pointer arithmetic
        out_f32 = torch.empty((B, S, K), dtype=torch.float32, device=x.device)

        # Dynamic tiling for gating based on K
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

        # Cast back to bfloat16 to match original behavior and return
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
