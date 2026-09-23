import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row (flattened [B*S]).
    x_ptr: flattened input pointer, stride per row is K (row base = i*K).
    mean_ptr, std_ptr: shape [total_rows].
    """
    row = tl.program_id(0)  # 0..total_rows-1
    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate across K in chunks of BLOCK_SIZE
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    std = tl.sqrt(var)  # population std, matches torch.std(..., unbiased=False)

    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity,
                          a1, a2, a3, a4, a5, a6,
                          b1, b2, b3, b4, b5,
                          c1, c2, c3, c4, c5, c6,
                          d1, d2, d3, d4,
                          p_low, p_high,
                          num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) z for a given p via A&S 5.2.23.
    Write result into z_buf[0] as fp32 scalar.
    """
    # This kernel runs with a single program; we pass target_sparsity and compute z.
    p = target_sparsity  # scalar in [0,1]
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p > (1.0 - p_low):
        # Upper region
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Write z to z_buf[0]
    tl.store(z_buf, result)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply elementwise gating: out = relu(x - (mean + std * z))
    Grid: (rows, column_tiles). Each program handles one row and one column tile.
    """
    row = tl.program_id(0)  # 0..total_rows-1
    col_tile = tl.program_id(1)  # 0..column_tiles-1
    offs = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_ptr)  # scalar

    # Load input chunk
    x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute gating
    thresh = mean + std * z
    y = x - thresh
    y = tl.maximum(y, 0.0)  # ReLU

    # Store
    tl.store(out_ptr + row * K + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [B, S, K] tensor (any dtype), CUDA device
        Returns: bfloat16 tensor [B, S, K] after adaptive threshold gating.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure CUDA tensors and contiguous layout
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x_contig = x.contiguous()
        B, S, K = x_contig.shape
        total_rows = B * S

        # Allocate fp32 buffers for mean and std (shape [B, S, 1] to match original)
        mean = torch.empty((B, S, 1), dtype=torch.float32, device=x_contig.device)
        std = torch.empty((B, S, 1), dtype=torch.float32, device=x_contig.device)

        # 1) Compute per-row mean and std
        # Choose a reasonable BLOCK_SIZE for reduction; K may vary, but this is fine for large K
        block_size_reduce = 2048
        num_warps_reduce = 4
        compute_row_stats_kernel[(total_rows,)](
            x_contig.view(-1), mean.view(-1), std.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_reduce,
            num_warps=num_warps_reduce,
            num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        z_buf = torch.empty(1, dtype=torch.float32, device=x_contig.device)

        # Constants for A&S 5.2.23
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
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # read scalar without torch tensor creation on host

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x_contig.to(torch.float32)  # compute in fp32 for stability
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
            x_f32.view(-1), mean.view(-1), std.view(-1), z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
