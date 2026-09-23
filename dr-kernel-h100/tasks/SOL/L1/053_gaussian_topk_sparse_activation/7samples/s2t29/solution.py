import torch
import triton
import triton.language as tl


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a scalar p using Abramowitz & Stegun 7.1.26 approximation.
    Writes the result into out_ptr[0] as float32.
    """
    # Constants
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

    # Local scalar math
    p = float(p)  # ensure fp32
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # Horner's method for polynomial
        poly = c1
        poly = poly * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1
        denom = denom * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        result = poly / denom
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        poly = a1
        poly = poly * r + a2
        poly = poly * r + a3
        poly = poly * r + a4
        poly = poly * r + a5
        poly = poly * r + a6
        poly = poly * q
        denom = b1
        denom = denom * r + b2
        denom = denom * r + b3
        denom = denom * r + b4
        denom = denom * r + b5
        denom = denom * r + 1.0
        result = poly / denom
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = c1
        poly = poly * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1
        denom = denom * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        result = -poly / denom

    tl.store(out_ptr, result)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_ptr, out_ptr,
                 rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over a 2D view [rows, N]:
    out[row, col] = max(0, x[row, col] - (mean[row] + std[row] * inv_ptr[0]))
    x_ptr/out_ptr are pointers to [rows, N] flattened row-major.
    mean_ptr/std_ptr are pointers to [rows].
    inv_ptr is a 1-element tensor containing inv_norm_cdf(target_sparsity).
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load scalars for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_ptr)  # scalar

    # Load input row slice
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    threshold = mean + std * inv_cdf
    y = tl.maximum(x - threshold, 0.0)  # ReLU(x - threshold)
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized sparse activation:
    - Compute mean and std along the feature dimension using PyTorch.
    - Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel.
    - Apply gating via Triton elementwise kernel.
    Returns bfloat16 tensor with same shape as inputs.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Compute statistics in fp32 for stability
    inputs_f32 = inputs.to(torch.float32)
    B, S, N = inputs_f32.shape
    # Mean and std along last dim (feature)
    mean = torch.mean(inputs_f32, dim=-1)            # [B, S]
    std = torch.std(inputs_f32, dim=-1, unbiased=False)  # [B, S]
    # Expand to [B, S, 1] for broadcasting across feature dim
    mean = mean.unsqueeze(-1)
    std = std.unsqueeze(-1)

    # Prepare 2D view [rows, N]
    rows = B * S
    x_2d = inputs_f32.view(rows, N)
    mean_2d = mean.view(rows, 1).expand(rows, N)     # broadcast to [rows, N]
    std_2d = std.view(rows, 1).expand(rows, N)

    # Compute inv_norm_cdf(target_sparsity) using Triton scalar kernel
    inv_cdf_buf = torch.empty(1, device=inputs.device, dtype=torch.float32)
    compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

    # Apply gating via Triton 2D kernel over tiles of N
    out_2d = torch.empty((rows, N), device=inputs.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean_2d, std_2d, inv_cdf_buf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back to [B, S, N] and cast to bfloat16
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor: [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise ValueError("ModelNew expects a single input tensor")
        return run(args[0])


def run(*args):
    return ModelNew()(*args)
