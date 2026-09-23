import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std across the last dim K for each row.
    x_ptr is a flattened view of shape [total_rows, K], row stride = K.
    """
    row_id = tl.program_id(0)
    # Bounds check: if row_id >= total_rows, return
    # (grid is set to total_rows, so no need here, but kept for safety if extended)
    # Initialize accumulators
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # Loop over columns in tiles of BLOCK_SIZE
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Address for this row and column block
        x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
        # Accumulate sums in fp32
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    # Compute mean and std (population std, unbiased=False)
    K_fp = tl.cast(K, tl.float32)
    mean = sum_x / K_fp
    var = sum_x2 / K_fp - mean * mean
    # Ensure non-negative variance before sqrt (numerical safety)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(out_ptr, p: tl.float32):
    """
    Compute inverse standard normal CDF (quantile function) for p in (0, 1).
    Uses Abramowitz and Stegun 5.2.23 approximation.
    out_ptr: 1-element buffer to write result.
    """
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

    # Lower region: x < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    r = q * q
    poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    lower = poly / den

    # Central region: p_low <= p <= p_high
    q = p - 0.5
    r = q * q
    poly_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    central = poly_mid / den_mid

    # Upper region: p > p_high
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    r = q * q
    poly_up = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den_up = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    upper = -poly_up / den_up

    # Piecewise selection
    result = tl.where(p < p_low, lower, tl.where(p > p_high, upper, central))
    tl.store(out_ptr, result)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K,
                           BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = max(x - (mean + std * z), 0) elementwise.
    Grid is (total_rows, cdiv(K, BLOCK_SIZE)).
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    col_start = tile_id * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load scalar z and per-row mean/std
    z = tl.load(z_ptr)  # scalar
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)

    # Compute threshold per row
    threshold = mean + std * z

    # Load x row segment
    x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
    # Gate: max(x - threshold, 0)
    gated = x - threshold
    gated = tl.maximum(gated, 0.0)
    tl.store(out_ptr + row_id * K + offs, gated, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation of the original logic:
        1) Compute per-row mean and std across last dim (population std).
        2) Compute z = _ndtri(target_sparsity) in Triton.
        3) Apply gating: out = relu(x - (mean + std * z)).
        Returns: tensor in bfloat16, same shape as input.
        """
        # Ensure contiguous and CUDA
        if not x.is_cuda:
            x = x.contiguous().to("cuda")
        x_contig = x.contiguous()

        # Flatten to [total_rows, K], where total_rows = B * S
        total_rows = x_contig.shape[0] * x_contig.shape[1]
        K = x_contig.shape[2]

        # Allocate mean/std buffers (fp32 for compute)
        mean = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)

        # 1) Compute per-row stats
        # Choose BLOCK_SIZE for reduction
        if K >= 16384:
            BLOCK_SIZE = 4096
            num_warps = 4
        elif K >= 4096:
            BLOCK_SIZE = 2048
            num_warps = 4
        else:
            BLOCK_SIZE = 1024
            num_warps = 4

        x_flat = x_contig.view(-1)  # shape [total_rows * K]
        grid_stats = (total_rows,)
        compute_row_stats_kernel[grid_stats](
            x_flat, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (single scalar)
        z_buf = torch.empty(1, dtype=torch.float32, device=x_contig.device)
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            num_warps=1,
            num_stages=1
        )
        z = float(z_buf.item())  # read scalar without torch tensor creation on host

        # 3) Apply gating in Triton with 2D tiling; compute in fp32, output will be cast later
        x_f32 = x_contig.to(torch.float32)  # compute in fp32 for numerical stability
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
