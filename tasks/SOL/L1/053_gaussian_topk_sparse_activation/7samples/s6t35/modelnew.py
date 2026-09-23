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
    row_start = (b * S + s) * F
    # Accumulate sum and sum of squares across features
    total = tl.zeros((), dtype=tl.float32)
    total2 = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    # Loop over feature dimension in chunks
    for start in range(0, F, BLOCK):
        idx = start + offs
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)
    n = F
    mean = total / n
    var = total2 / n - mean * mean
    std = tl.sqrt(var)
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT[b,s,f] = relu(X[b,s,f] - (MEANS[b*s] + STDs[b*s] * ICDF)),
    where ICDF is a scalar (inverse normal CDF for target sparsity).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    cutoff = mean + std * ICDF
    offs = tl.arange(0, BLOCK)
    for start in range(0, F, BLOCK):
        idx = start + offs
        mask = idx < F
        x = tl.load(X + row_start + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)
        tl.store(OUT + row_start + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(PROB, OUT):
    """
    Compute icdf(prob) using Abramowitz & Stegun 5.2.23 central region approximation.
    Prob: 1-element tensor (float32), OUT: 1-element tensor (float32) output.
    """
    p = tl.load(PROB)
    # Central region approximation (q = p - 0.5)
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
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly * q / den
    tl.store(OUT, icdf)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        - Compute rowwise mean and std across last dim
        - Compute icdf(target_sparsity) using A&S 5.2.23 approximation in Triton
        - Apply ReLU(x - (mean + std * icdf)) and return in bfloat16
        """
        if target_sparsity == 0.0:
            return x

        # Ensure fp32 for numeric stability in reduction
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Allocate per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Choose BLOCK and num_warps based on F (compile-time constants for Triton)
        if F >= 2048:
            BLOCK_STATS = BLOCK_SP = 2048
            num_warps_stats = 8
            num_warps_sp = 8
        else:
            BLOCK_STATS = BLOCK_SP = 1024
            num_warps_stats = 4
            num_warps_sp = 4

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=3
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        return out.to(torch.bfloat16)