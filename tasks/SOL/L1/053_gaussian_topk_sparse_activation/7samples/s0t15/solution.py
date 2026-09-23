import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row flattened to [total_rows].
    x_ptr points to a contiguous [total_rows, K] view (i.e., stride_row == K).
    mean_ptr, std_ptr are 1D arrays of length total_rows.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over the feature dimension K in tiles
    for offs in range(0, K, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # x_ptr is treated as [total_rows, K] contiguous; row stride is K
        vals = tl.load(x_ptr + row_id * K + cols, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        acc_sum += tl.sum(vals_f32, axis=0)
        acc_sumsq += tl.sum(vals_f32 * vals_f32, axis=0)

    mean = acc_sum / K
    # population std: sqrt(E[x^2] - (E[x])^2)
    std = tl.sqrt(acc_sumsq / K - mean * mean)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high):
    """
    Compute inverse standard normal CDF z = ndtri(p) using Abramowitz & Stegun 5.2.23 approximation.
    Stores result in z_buf_ptr[0].
    p: scalar float in (0, 1). We pass target_sparsity (host) as a float.
    """
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
    den_mid = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    z_mid = poly_mid * q_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1*q_high + c2)*q_high + c3)*q_high + c4)*q_high + c5)*q_high + c6) / ((((d1*q_high + d2)*q_high + d3)*q_high + d4)*q_high + 1.0)

    # Select based on p region
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    # default to mid
    z_val = z_mid
    z_val = tl.where(cond_low, z_low, z_val)
    z_val = tl.where(cond_mid, z_mid, z_val)  # mid already
    z_val = tl.where(~cond_low & ~cond_mid, z_high, z_val)

    # Store to z_buf_ptr[0]
    tl.store(z_buf_ptr, z_val)


@triton.jit
def apply_gating_2d_kernel(x_flat_ptr, mean_flat_ptr, std_flat_ptr, z_ptr, out_flat_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out[row, col] = relu(x[row, col] - (mean[row] + std[row] * z)),
    where z = z_ptr[0] (scalar). x_flat_ptr points to a contiguous [total_rows, K] view.
    """
    row_id = tl.program_id(0)
    col_tile = tl.program_id(1)
    if row_id >= total_rows:
        return

    col_start = col_tile * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load row stats
    mean = tl.load(mean_flat_ptr + row_id)
    std = tl.load(std_flat_ptr + row_id)
    z = tl.load(z_ptr)  # scalar

    # Load input row slice
    vals = tl.load(x_flat_ptr + row_id * K + cols, mask=mask, other=0.0)
    # Compute gating
    threshold = mean + std * z
    gated = vals - threshold
    gated = tl.maximum(gated, 0.0)  # ReLU
    tl.store(out_flat_ptr + row_id * K + cols, gated, mask=mask)


def _choose_gate_params(K: int):
    # Heuristic tuning for gating kernel
    if K >= 8192:
        return 4096, 8, 2  # BLOCK_SIZE, num_warps, num_stages
    else:
        return 2048, 4, 2


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure we are on CUDA for Triton; if not, fallback to a safe PyTorch path
        if not inputs.is_cuda:
            # Fallback: keep original behavior, but run on CPU (Triton requires CUDA)
            x_f32 = inputs.to(torch.float32)
            # Compute stats
            mean = x_f32.mean(dim=-1, keepdim=True)
            # torch.std(..., unbiased=False) for population std
            std = x_f32.std(dim=-1, keepdim=True, unbiased=False)
            # Inverse normal CDF via PyTorch
            z = torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
            cutoff = mean + std * z
            sparse = torch.relu(x_f32 - cutoff)
            return sparse.to(torch.bfloat16)

        # Triton-only path
        # Flatten rows: [B*S, K]
        total_rows = inputs.shape[0] * inputs.shape[1]
        K = inputs.shape[2]
        x_flat = inputs.contiguous().view(total_rows, K)

        # Allocate stats buffers (float32 for compute)
        mean = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)

        # Choose reduction tile size
        # Use a moderate BLOCK_SIZE for reduction; 1024 works well across a range
        BLOCK_SIZE_STATS = 1024
        num_warps_stats = 4

        # Launch reduction kernel
        grid_stats = (total_rows,)
        compute_row_stats_kernel[grid_stats](
            x_flat, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=num_warps_stats,
            num_stages=2
        )

        # Compute z = ndtri(target_sparsity) via Triton kernel
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        # Constants for A&S approximation
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
            num_warps=1, num_stages=1
        )

        # Prepare input and output for gating
        x_f32 = x_flat.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tiling for gating
        block_size_gate, num_warps_gate, num_stages_gate = _choose_gate_params(K)
        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))

        # Launch gating kernel
        apply_gating_2d_kernel[grid_gate](
            x_f32, mean, std, z_buf, out_f32,
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=num_stages_gate
        )

        # Return in bfloat16
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
