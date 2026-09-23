import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, MEANS, STDs, B, S, F, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), linearized as [B*S, F]
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    # Compute row start offset in flattened [B*S, F] view
    row_start = pid * F

    sum_val = 0.0
    sum_sq = 0.0

    i = 0
    while i < F:
        idx = row_start + i + tl.arange(0, BLOCK)
        mask = i + tl.arange(0, BLOCK) < F
        x = tl.load(X + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    mean = sum_val / F
    # Population std (unbiased=False)
    var = sum_sq / F - mean * mean
    std = tl.sqrt(var)

    # Store results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, icdf, Y, B, S, F, BLOCK: tl.constexpr):
    """
    Apply ReLU(X - (mean + std * icdf)) per row, write to Y.
    X: [B, S, F] input (float32), linearized as [B*S, F]
    MEANS: [B*S] per-row mean (float32)
    STDs: [B*S] per-row std (float32)
    icdf: scalar (float32) inverse normal CDF for target sparsity
    Y: [B, S, F] output (float32), same layout
    """
    pid = tl.program_id(axis=0)
    row_start = pid * F

    # Load per-row stats
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    threshold = mean + std * icdf

    i = 0
    while i < F:
        idx = row_start + i + tl.arange(0, BLOCK)
        mask = i + tl.arange(0, BLOCK) < F
        x = tl.load(X + idx, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y + idx, y, mask=mask)
        i += BLOCK


@triton.jit
def _icdf_ndtri_kernel(P, Y, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for standard normal for scalar P in (0, 1).
    Uses Abramowitz & Stegun 5.2.23 central region approximation.
    Writes result into Y (single element).
    """
    # Load probability p
    p = tl.load(P)
    # Central region: p in [0.5, 1]
    q = p - 0.5
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as per A&S
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    y = poly * q / denom  # central region

    # Store result (single element)
    tl.store(Y, y)


def _run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Computes per-row mean and std along last dim (features).
    - Computes threshold = mean + std * icdf(target_sparsity).
    - Applies ReLU(input - threshold).
    Returns tensor in bfloat16.
    """
    # Expect input of shape [batch_size, seq_len, intermediate_size]
    x = inputs
    B, S, F = x.shape

    # Compute in float32 for numerical stability
    x_fp32 = x.to(torch.float32)

    # Allocate stats buffers (flattened rows)
    means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

    # Launch rowwise stats kernel: one program per (b, s) row
    BLOCK_STATS = 1024
    grid = (B * S,)
    _rowwise_stats_kernel[grid](
        x_fp32, means, stds, B, S, F, BLOCK_STATS,
        num_warps=4, num_stages=2
    )

    # Compute icdf for target_sparsity using Triton (single scalar)
    icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
    )

    # Output buffer (float32 for compute)
    out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

    # Launch sparsify + ReLU kernel
    BLOCK_SP = 1024
    _sparsify_relu_kernel[grid](
        x_fp32, means, stds, icdf, out_fp32, B, S, F, BLOCK_SP,
        num_warps=4, num_stages=2
    )

    # Return in bfloat16, matching original behavior
    return out_fp32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Match original Model's signature: expects one input tensor
        return _run(args[0], 0.0)