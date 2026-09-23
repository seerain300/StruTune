import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std (unbiased=False) across the last dimension.
    x_ptr: pointer to input [rows, N]
    mean_ptr/std_ptr: output vectors of length rows (float32)
    N: last dimension size (int32)
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Row-major: index = row_id * N + offs
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_kernel(out_ptr, p: tl.float32, BLOCK_SIZE: tl.constexpr):
    """
    Compute inv_norm_cdf(p) using Abramowitz & Stegun approximation (formula 26.2.23).
    Writes a single scalar to out_ptr.
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

    # Vector pvec of length 1
    pvec = tl.full((BLOCK_SIZE,), p, tl.float32)
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    mask_low = pvec < p_low
    q_low = tl.sqrt(-2.0 * tl.log(pvec[mask_low]))
    result[mask_low] = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                       (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))

    mask_mid = (pvec >= p_low) & (pvec <= p_high)
    q_mid = pvec[mask_mid] - 0.5
    r_mid = q_mid * q_mid
    result[mask_mid] = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                       (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    mask_high = pvec > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - pvec[mask_high]))
    result[mask_high] = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                        ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    tl.store(out_ptr, result[0])


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    2D grid kernel: axis=0 over rows, axis=1 over column tiles.
    y = max(0, x - (mean + std * inv_cdf))
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    # Load input row tile
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Compute gated output
    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.where(y > 0.0, y, 0.0)

    # Store to output (out is 1D flattened: index = row_id * N + offs)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [batch_size, seq_len, intermediate_size]
        target_sparsity: float in (0, 1)
        Returns: bfloat16 tensor with gated activations.
        """
        # Ensure device and dtype; compute in float32 for numerical stability
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels"
        x_f32 = x.to(torch.float32)
        B, S, N = x_f32.shape
        rows = B * S

        # Allocate mean and std buffers (per row)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Compute inv_norm_cdf(target_sparsity) via Triton
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri_kernel[(1,)](inv_cdf_buf, p=float(target_sparsity), BLOCK_SIZE=1)

        # Prepare 2D input/output views [rows, N]
        x_2d = x_f32.view(rows, N)
        out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)

        # 1) Compute per-row mean and std using Triton reduction
        BLOCK_SIZE_RS = 1024  # good default; loop covers N
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Apply gating with Triton 2D kernel
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid_gt = (rows, num_tiles)
        gate_rows_2d[grid_gt](x_2d, mean, std, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

        # 3) Cast to bfloat16 and return
        out = out_2d.view(B, S, N).to(torch.bfloat16)
        return out