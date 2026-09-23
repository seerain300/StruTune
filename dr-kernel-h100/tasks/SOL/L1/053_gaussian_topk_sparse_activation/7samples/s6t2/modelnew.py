import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X_ptr, B, S, F, MEAN_ptr, STD_ptr, BLOCK: tl.constexpr):
    """
    For each (b, s) row in X, compute mean and std over last dimension (length F).
    Accumulate sum and sum of squares in fp32.
    """
    row_id = tl.program_id(0)  # 0..(B*S - 1)
    # Map row_id to (b, s)
    b = row_id // S
    s = row_id % S
    # Base pointer for this row (since we call this on a 2D view of X: rows = B*S, cols = F)
    base = b * S * F + s * F  # row offset in contiguous [B*S, F]

    sum_val = 0.0
    sum_sq = 0.0

    off = 0
    while off < F:
        offs = off + tl.arange(0, BLOCK)
        mask = offs < F
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # masked load ensures out-of-bounds elements are zero
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        off += BLOCK

    n = F  # population std
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # guard against negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # store results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


@triton.jit
def _icdf_approx_kernel(P_ptr, ICDF_ptr, BLOCK: tl.constexpr):
    """
    Compute inverse standard normal CDF for a single scalar probability using
    Abramowitz & Stegun 5.2.23 approximation and store into ICDF_ptr[0].
    Assumes P_ptr[0] contains the target sparsity scalar (0..1).
    """
    # There is only one program instance, computing the scalar icdf.
    p = tl.load(P_ptr + 0).to(tl.float32)

    # Clamp p to [0, 1] to avoid log(0) or log(1) issues
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

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

    # Region masks
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region: p < p_low
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        icdf = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        icdf = -icdf  # inverse CDF is negative in lower tail

    # Central region: p_low <= p <= p_high
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        icdf = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region: p > p_high
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        icdf = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(ICDF_ptr + 0, icdf)


@triton.jit
def _sparsify_relu_kernel(X_ptr, B, S, F, MEAN_ptr, STD_ptr, ICDF_ptr, OUT_ptr, BLOCK: tl.constexpr):
    """
    For each (b, s) row, load mean and std, compute threshold = mean + std * ICDF,
    then apply ReLU(X - threshold) over last dimension and store fp32 output.
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    base = b * S * F + s * F

    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    icdf = tl.load(ICDF_ptr + 0)

    threshold = mean + std * icdf

    off = 0
    while off < F:
        offs = off + tl.arange(0, BLOCK)
        mask = offs < F
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + base + offs, y, mask=mask)
        off += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float = 0.0) -> torch.Tensor:
        """
        Triton-only implementation of the original run behavior:
        - Compute per (batch, seq) row mean and std over last dim.
        - Compute inverse normal CDF for target_sparsity using Triton.
        - Apply ReLU(input - (mean + std * icdf)) and return bfloat16.
        """
        # Ensure 3D input [B, S, F]
        assert inputs.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, F = inputs.shape

        # Compute in fp32 for numerical stability
        x_fp32 = inputs.to(torch.float32)

        # Allocate mean and std buffers for rows (B*S,)
        means = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)

        # Launch rowwise stats kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32,
            B, S, F,
            means, stds,
            BLOCK=1024,
            num_warps=4,
        )

        # Compute icdf for the single scalar target_sparsity using Triton (no host torch ops)
        p = torch.tensor([target_sparsity], dtype=torch.float32, device=inputs.device)
        icdf = torch.empty((1,), dtype=torch.float32, device=inputs.device)
        _icdf_approx_kernel[(1,)](
            p,
            icdf,
            BLOCK=1,
            num_warps=1,
        )

        # Allocate output fp32
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs.device)

        # Launch sparsification kernel: one program per (b, s) row
        _sparsify_relu_kernel[grid](
            x_fp32,
            B, S, F,
            means, stds, icdf, out_fp32,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bfloat16, matching original behavior
        return out_fp32.to(torch.bfloat16)