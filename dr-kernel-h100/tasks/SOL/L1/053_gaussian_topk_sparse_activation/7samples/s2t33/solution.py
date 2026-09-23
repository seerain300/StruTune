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
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, out_ptr, N, target_sparsity, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating over tiles:
    For each row, compute threshold = mean + std * inv_norm_cdf(target_sparsity) and
    write y = max(0, x - threshold) to out_ptr.
    x_ptr/out_ptr: [rows, N] contiguous
    mean_ptr/std_ptr: [rows] scalars
    N: number of columns
    target_sparsity: float scalar probability in (0, 1)
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    # Compute inverse standard normal CDF using Abramowitz & Stegun approximation (formula 7.1.26)
    # This is a per-program scalar; same for all columns in the row.
    p = target_sparsity
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Coefficients for A&S approximation
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

    # lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_low = poly_low / (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    inv_low = -poly_low

    # central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    poly_mid = poly_mid / (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    poly_up = poly_up / (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
    inv_up = poly_up

    # select appropriate branch
    if p < p_low:
        inv_cdf = inv_low
    elif p <= p_high:
        inv_cdf = poly_mid
    else:
        inv_cdf = inv_up

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    threshold = mean + std * inv_cdf  # scalar

    # Process tile of columns
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    # y = max(0, x - threshold)
    y = x - threshold
    y = tl.where(y > 0.0, y, 0.0)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Emulate the original: run(inputs, target_sparsity)
        # If no inputs, return None (optional). We assume a single input tensor and optional sparsity.
        if len(args) == 0:
            return None
        inputs = args[0]
        target_sparsity = 0.5  # default; if provided, original would expect a second arg
        if len(args) > 1:
            target_sparsity = float(args[1])

        # Compute in float32 for stability; keep original shape
        x = inputs
        x_f32 = x.to(torch.float32)

        # Flatten to 2D [rows, N] where N = last dim
        N = x_f32.shape[-1]
        rows = x_f32.numel() // N
        x_2d = x_f32.view(rows, N)

        # 1) Triton reduction to compute per-row mean and std (unbiased=False)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 1024
        reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Triton elementwise gating over tiles (computes inv_norm_cdf inside the kernel)
        out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_2d[grid](x_2d, mean, std, out_2d, N, target_sparsity, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

        # Reshape back to original shape and cast to bfloat16 to match original behavior
        out = out_2d.view(*x.shape).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
