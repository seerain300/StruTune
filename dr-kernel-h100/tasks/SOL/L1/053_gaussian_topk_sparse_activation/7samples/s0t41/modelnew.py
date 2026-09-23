import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row (flattened [B*S]).
    Assumes x_ptr points to a contiguous [total_rows, K] layout (row-major).
    """
    pid = tl.program_id(0)
    # Each program handles one row (pid in [0, total_rows))
    row_start = pid * K

    # Accumulate sum and sum of squares in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Iterate over columns in tiles
    for col in range(0, K, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # Load a tile of the row
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Reduce within the tile
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = tl.float32(K)
    mean = sum_val / n
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / n - mean * mean
    # ensure non-negative due to numerical errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low):
    """
    Compute inverse standard normal CDF at p using Abramowitz & Stegun 5.2.23 approximation.
    Writes a single scalar into z_buf_ptr (float32).
    """
    # Constants
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Masks for regions
    # In Triton, we implement piecewise logic via masks; only one region will be active per lane.
    # We'll compute all branches and write only once (masked).
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z_low = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    if (p >= p_low) & (p <= p_high):
        q = p - 0.5
        r = q * q
        z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        z_mid = 0.0

    # Choose based on p
    # Triton doesn't support dynamic branching on scalar easily; use masks:
    out = tl.where(p < p_low, z_low, tl.where(p <= p_high, z_mid, 0.0))

    # Store the scalar
    tl.store(z_buf_ptr, out)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z)), with broadcasting over columns per row.
    x_ptr points to input [total_rows*K], mean_ptr and std_ptr are [total_rows], z_ptr is scalar.
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < total_rows) & (cols < K)

    # Load mean and std for the row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_ptr)  # scalar

    # Compute threshold
    threshold = mean + std * z

    # Compute input offsets for this row
    row_start = row * K
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store
    tl.store(out_ptr + row_start + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation with adaptive threshold:
        - Compute per-row mean and std over last dim (K).
        - Compute z = inverse normal CDF at target_sparsity using A&S approximation.
        - Threshold per row: mean + std * z.
        - Output = relu(x - threshold) in bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous [B, S, K]
        x = inputs.contiguous()
        B, S, K = x.shape
        total_rows = B * S

        # Prepare contiguous view for reduction
        x_contig = x.view(total_rows, K)

        # Allocate outputs for mean and std
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Launch reduction kernel (fp32 accumulators)
        # Heuristics: choose BLOCK_SIZE based on K for better occupancy
        if K >= 16384:
            BLOCK_SIZE = 4096
        elif K >= 8192:
            BLOCK_SIZE = 2048  # safer smaller tile to reduce reduction pressure
        else:
            BLOCK_SIZE = 1024

        grid = (total_rows,)
        compute_row_stats_kernel[grid](
            x_contig, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2 if K >= 8192 else 1
        )

        # Compute z = _ndtri(target_sparsity) via Triton approximation
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        # Constants for A&S approximation
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

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low,
            num_warps=1, num_stages=1
        )

        z = float(z_buf.item())  # read single scalar without torch tensor creation

        # Apply gating with 2D Triton kernel
        x_f32 = x.to(torch.float32)  # compute in fp32 for numerical stability
        out_f32 = torch.empty_like(x_f32)

        # Deterministic tiling based on K
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
            num_stages_gate = 2
        elif K >= 4096:
            block_size_gate = 2048
            num_warps_gate = 4
            num_stages_gate = 2
        else:
            block_size_gate = 1024
            num_warps_gate = 4
            num_stages_gate = 1

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=num_stages_gate
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)