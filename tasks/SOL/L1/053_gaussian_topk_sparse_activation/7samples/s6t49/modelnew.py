import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32)
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base offset for the row
    row_offset = (b * S + s) * F
    row_sum = 0.0
    row_sumsq = 0.0

    # Iterate over the feature dimension in BLOCK-sized chunks
    for offset in range(0, F, BLOCK):
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_offset + idx, mask=mask, other=0.0)
        row_sum += tl.sum(x, axis=0)
        row_sumsq += tl.sum(x * x, axis=0)

    n = F
    mean = row_sum / n
    # Population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = row_sumsq / n - mean * mean
    # Ensure non-negative due to numerical precision
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write out
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, Y, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: Y = relu(X - (MEANS + STDs * ICDF))
    MEANS and STDs are per row scalars; ICDF is a scalar.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_offset = (b * S + s) * F
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    cutoff = mean + std * ICDF
    for offset in range(0, F, BLOCK):
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + row_offset + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(Y + row_offset + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(P, Y, BLOCK: tl.constexpr):
    """
    Compute inverse standard normal CDF for p using Abramowitz & Stegun 5.2.23 central-region approximation.
    Y is a 1-element tensor to store the result.
    """
    # Load probability p
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    # Coefficients
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # note: negative, important for correctness

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Central region polynomial
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    y = poly / denom

    # Store result
    tl.store(Y, y)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    Computes adaptive sparsity threshold based on input statistics:
      threshold = mean + std * norm.icdf(target_sparsity)
    Then applies ReLU(input - threshold).

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity.

    Returns:
        Sparsified tensor of same shape as input.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and compute in float32
    x = inputs.contiguous()
    x_fp32 = x.to(torch.float32)

    B, S, F = x_fp32.shape

    # Output buffer in float32
    out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

    # Per-row means and stds
    means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

    # Fixed BLOCK for robustness
    BLOCK = 1024

    # Launch rowwise stats kernel
    grid = (B * S,)
    _rowwise_stats_kernel[grid](
        x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=4, num_stages=2
    )

    # Compute icdf for target_sparsity using Triton (single scalar)
    icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
    )

    # Launch sparsify + ReLU kernel
    _sparsify_relu_kernel[grid](
        x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=4, num_stages=2
    )

    # Return in bfloat16, matching original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)