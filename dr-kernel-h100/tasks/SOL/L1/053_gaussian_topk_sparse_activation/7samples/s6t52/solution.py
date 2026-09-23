import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, MEANS, STDs, F, BLOCK: tl.constexpr):
    """
    Compute per-row mean and population std across last dimension (features).
    X: [B*S, F] input (float32), flattened rows
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    # Each program handles one row: pid in [0, B*S)
    row_start = pid * F
    offs = tl.arange(0, BLOCK)

    sum_val = 0.0
    sum_sq = 0.0

    # Loop over the features in chunks of BLOCK
    for i in range(0, F, BLOCK):
        idx = i + offs
        mask = idx < F
        # Load a chunk of the row; masked loads for tails
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        # Accumulate sum and sum of squares in fp32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    # Population std (unbiased=False): sqrt(E[x^2] - mean^2)
    var = sum_sq / n - mean * mean
    # Clamp var to >= 0 to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a scalar probability P using Abramowitz & Stegun 5.2.23 approximation.
    Uses central region only: q = p - 0.5.
    OUT: 1-element tensor to store icdf
    """
    # Load p
    p = tl.load(P)
    # Central region approximation
    q = p - 0.5
    r = q * q

    # Coefficients (float64 for better precision, then cast to float32)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # note: negative as per A&S
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Compute numerator and denominator in fp64
    num = (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    )
    den = (
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )
    icdf = num / den

    # Store as fp32
    tl.store(OUT, icdf.to(tl.float32))


@triton.jit
def _sparsify_relu_kernel(IN, MEANS, STDs, ICDF, OUT, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = max(0, IN - (MEANS + STDs * ICDF)), elementwise per row.
    IN: [B*S, F] input rows (float32)
    MEANS: [B*S] per-row means (float32)
    STDs: [B*S] per-row stds (float32)
    ICDF: 1-element tensor (float32)
    OUT: [B*S, F] output rows (float32)
    """
    pid = tl.program_id(axis=0)
    row_start_in = pid * F
    row_start_out = pid * F
    offs = tl.arange(0, BLOCK)
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    multiplier = tl.load(ICDF)
    thresh = mean + std * multiplier

    # Process the row in chunks
    for i in range(0, F, BLOCK):
        idx = i + offs
        mask = idx < F
        x = tl.load(IN + row_start_in + idx, mask=mask, other=0.0)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT + row_start_out + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA tensors and use float32 for compute
        device = x.device
        dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape
        # Flatten rows to [B*S, F] for simpler indexing in Triton
        x_rows = x_fp32.view(B * S, F)
        out_rows = torch.empty((B * S, F), dtype=torch.float32, device=device)

        # Per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Choose BLOCK and execution parameters
        if F >= 2048:
            BLOCK_STATS = 2048
            num_warps_stats = 8
            num_stages_stats = 3
        else:
            BLOCK_STATS = 1024
            num_warps_stats = 4
            num_stages_stats = 2

        # Launch rowwise stats kernel
        grid_stats = (B * S,)
        _rowwise_stats_kernel[grid_stats](
            x_rows, means, stds, F, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=device)
        p_dev = torch.empty((), dtype=torch.float32, device=device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        if F >= 2048:
            BLOCK_SP = 2048
            num_warps_sp = 8
            num_stages_sp = 3
        else:
            BLOCK_SP = 1024
            num_warps_sp = 4
            num_stages_sp = 2

        _sparsify_relu_kernel[grid_stats](
            x_rows, means, stds, icdf, out_rows, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Reshape back to [B, S, F] and return in bfloat16
        out = out_rows.view(B, S, F).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
