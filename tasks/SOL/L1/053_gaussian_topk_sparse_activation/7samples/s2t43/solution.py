import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Row offset is row_id * N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_scalar(p_ptr, inv_c_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel: compute inverse normal CDF (quantile) for p[0] using A&S approximation.
    Writes result to inv_c_ptr[0] as float32.
    """
    # Load p
    p = tl.load(p_ptr)
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
    p_high = 1.0 - p_low

    # Compute inverse for low region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    y_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # High region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select based on p
    inv_c = tl.where(p < p_low, y_low, 0.0)
    inv_c = tl.where((p >= p_low) & (p <= p_high), y_mid, inv_c)
    inv_c = tl.where(p > p_high, y_high, inv_c)

    tl.store(inv_c_ptr, inv_c)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_c, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating kernel:
    For each row, process tiles along N and compute y = max(0, x - (mean + std * inv_c)).
    x_ptr/out_ptr: [rows, N], contiguous row-major
    mean_ptr/std_ptr: [rows], float32
    inv_c: scalar float32 (passed as an argument)
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row's mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)

    # Load tile of x
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Compute threshold and gating
    threshold = mean + std * inv_c
    y = tl.maximum(x - threshold, 0.0)

    # Store result
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation of the Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold per row:
      threshold = mean + std * inv_norm_cdf(target_sparsity)
    Applies gating: y = max(0, x - threshold), returns bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Convert to float32 for numerical stability in statistics computation
    x = inputs.contiguous()
    x_f32 = x.to(torch.float32)

    # Flatten to 2D [rows, N], where N is the last dim
    B, S, N = x_f32.shape
    rows = B * S
    x_2d = x_f32.view(rows, N)

    # 1) Compute per-row mean and std (population, unbiased=False) in Triton
    mean = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) in Triton (single program)
    p_tensor = torch.tensor([float(target_sparsity)], device=x_2d.device, dtype=torch.float32)
    inv_c_tensor = torch.empty(1, device=x_2d.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](p_tensor, inv_c_tensor, BLOCK_SIZE=1, num_warps=1)

    inv_c = inv_c_tensor[0]  # scalar float32

    # 3) Apply gating via Triton 2D kernel over tiles of N
    out_2d = torch.empty((rows, N), device=x_2d.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_c, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back to [B, S, N] and cast to bfloat16
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input shaped [batch_size, seq_len, intermediate_size]
        if len(args) == 1:
            x = args[0]
        else:
            x = args[0]
        # Use a fixed target_sparsity as in the original signature; here 0.01
        return run(x, 0.01)


def run(*args):
    return ModelNew()(*args)
