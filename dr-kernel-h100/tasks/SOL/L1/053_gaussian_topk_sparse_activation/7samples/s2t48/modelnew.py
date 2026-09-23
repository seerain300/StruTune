import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_3d(x_ptr, mean_ptr, std_ptr, B, S, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim (N) for x of shape [B, S, N].
    x_ptr: pointer to input [B, S, N] contiguous
    mean_ptr/std_ptr: per-row outputs [B*S], float32
    """
    row_id = tl.program_id(axis=0)
    # Map row_id to (b, s)
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N

    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri_scalar(out_ptr, p, BLOCK_SIZE: tl.constexpr):
    """
    Triton scalar kernel: compute inverse standard normal CDF for scalar p in (0, 1).
    Uses Abramowitz and Stegun approximation (26.2.23).
    Writes to out_ptr[0].
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

    # Lower region
    # Note: Triton supports elementwise masking; here we evaluate q for all lanes and select via mask
    q_low = torch.sqrt(-2.0 * torch.log(p))  # This line is a placeholder; Triton kernel below has a scalar q
    # Compute via A&S logic for a scalar p
    # Implement A&S approximation here:
    if p < p_low:
        z = torch.sqrt(-2.0 * torch.log(p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        poly_over = (((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0))
        y = poly / poly_over
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly_over = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        y = (poly * q) / poly_over
    else:
        z = torch.sqrt(-2.0 * torch.log(1.0 - p))
        poly = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6)
        poly_over = (((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0))
        y = -poly / poly_over

    # Store to out_ptr[0]
    tl.store(out_ptr, y)


@triton.jit
def gate_rows_3d(x_ptr, out_ptr, mean_ptr, std_ptr, inv_cdf, B, S, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton elementwise gating over [B, S, N]: y = max(0, x - (mean + std * inv_cdf)).
    Grid: (rows=B*S, num_tiles_ceil(N/BLOCK_SIZE))
    """
    row_id = tl.program_id(axis=0)
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    threshold = mean + std * inv_cdf

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU equivalent
        tl.store(out_ptr + base + offs, y, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        1) Compute per-row mean and std across feature dim (last dim) for [B, S, N].
        2) Compute inv_norm_cdf(target_sparsity) using A&S approximation.
        3) Apply gating: y = max(0, x - (mean + std * inv_cdf)).
        Returns bfloat16 tensor of same shape as inputs.
        """
        # Handle trivial case
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and cast to float32 for computation
        x = inputs.to(torch.float32).contiguous()
        B, S, N = x.shape
        rows = B * S

        # 1) Compute per-row mean and std (population, unbiased=False)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE_RS = 256
        reduce_mean_std_3d[(rows,)](x, mean, std, B, S, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) via Triton scalar kernel
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        compute_inv_ndtri_scalar[(1,)](inv_cdf_buf, float(target_sparsity), BLOCK_SIZE=1)

        # 3) Apply gating via 3D Triton kernel over tiles of N
        out = torch.empty_like(x, device=x.device, dtype=torch.float32)
        BLOCK_SIZE_GT = 1024
        num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
        grid = (rows, num_tiles)
        gate_rows_3d[grid](x, out, mean, std, inv_cdf_buf[0], B, S, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=8)

        # Cast to bfloat16 to match original behavior
        out = out.to(torch.bfloat16)
        return out