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
    # Map pid to (b, s)
    b = pid // S
    s = pid % S

    # Accumulators (fp32)
    sum_row = 0.0
    sum_sq_row = 0.0

    # Iterate over features in blocks
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        # Linear indexing: X[b, s, idx]
        x = tl.load(X + b * S * F + s * F + idx, mask=mask, other=0.0)
        sum_row += tl.sum(x, axis=0)
        sum_sq_row += tl.sum(x * x, axis=0)

    mean = sum_row / F
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq_row / F - mean * mean
    std = tl.sqrt(var)

    # Store results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a given scalar probability P (0 < P < 1).
    Uses Abramowitz & Stegun 5.2.23 central region approximation.
    OUT: 1-element tensor to store icdf (float32)
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative, as per A&S
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly_num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly_num * q / poly_den

    tl.store(OUT, icdf)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, CUTOFF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT[b, s, f] = max(0, X[b, s, f] - (MEANS[b*s] + STDs[b*s] * CUTOFF))
    MEANS: [B*S]
    STDs: [B*S]
    CUTOFF: scalar float32
    OUT: [B, S, F]
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S * F + s * F

    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        mu = tl.load(MEANS + pid)
        sigma = tl.load(STDs + pid)
        threshold = mu + sigma * CUTOFF
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(OUT + base + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.

    Computes adaptive sparsity threshold based on input statistics:
    1) mean and std of input across feature dimension
    2) cutoff = mean + std * norm.icdf(target_sparsity)
    3) output = ReLU(input - cutoff)

    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity (all activations pass through).

    Returns:
        Sparsified tensor of same shape as input (bfloat16).
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure contiguous and compute in float32
    x = inputs.contiguous()
    x_fp32 = x.to(torch.float32)

    B, S, F = x_fp32.shape

    # Output buffer
    out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

    # Per-row means and stds
    means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

    # Fixed BLOCK for robustness and previously validated performance
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