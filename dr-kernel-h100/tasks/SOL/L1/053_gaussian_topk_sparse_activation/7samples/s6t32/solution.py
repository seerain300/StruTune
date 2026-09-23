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

    # Base index for the row (1D view [B*S, F])
    base = pid * F

    # Accumulators (fp32)
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over features in chunks of BLOCK
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    # Compute mean and population std (unbiased=False)
    mean = sum_x / F
    var = sum_x2 / F - mean * mean
    # Ensure non-negative due to numerical error
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_scalar_kernel(p_dev, icdf, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a single probability p using Abramowitz & Stegun 5.2.23 central region approximation.
    p_dev: [1] device tensor with scalar probability (float32)
    icdf: [1] device tensor to store result (float32)
    """
    # Load p (scalar)
    p = tl.load(p_dev)  # p in (0, 1)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q
    # Coefficients
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as per A&S 5.2.23
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly / den

    # Store icdf
    tl.store(icdf, z)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, icdf, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Elementwise sparsification for each row: OUT[b, s, f] = ReLU(X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * icdf))
    MEANS: [B*S], STDs: [B*S], icdf: scalar
    OUT: [B, S, F] (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Compute row mean and std from flattened arrays
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)

    # Load icdf scalar
    t = tl.load(icdf)  # multiplier = std * icdf

    # Base offset for this row (contiguous [B, S, F])
    base = pid * F

    # Loop over features
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        thr = mean + std * t
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: float in [0, 1]
        Returns: bfloat16 tensor of same shape after sparsification.
        """
        # Convert to float32 for compute
        x_fp32 = x.to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Launch rowwise stats kernel
        BLOCK_STATS = 1024
        grid_stats = (B * S,)
        _rowwise_stats_kernel[grid_stats](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=4, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_scalar_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Output buffer
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Launch sparsify + ReLU kernel
        BLOCK_SP = 1024
        grid_sp = (B * S,)
        _sparsify_relu_kernel[grid_sp](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=4, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
