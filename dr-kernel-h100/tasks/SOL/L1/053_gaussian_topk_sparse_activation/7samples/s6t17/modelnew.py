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

    # Base pointer to this row
    row_start = (b * S + s) * F
    row_ptr = X + row_start

    # Accumulators (fp32)
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in chunks
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        # Reduce within the block
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Mean and population std (unbiased=False)
    mean = sum_val / F
    var = sum_sq / F - mean * mean
    # Ensure non-negative due to numerical errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write outputs
    MEANS[pid] = mean
    STDs[pid] = std


@triton.jit
def _icdf_ndtri_kernel(p_dev, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a single scalar probability p.
    Uses Abramowitz & Stegun 5.2.23 central region formula.
    OUT[0] = icdf(p)
    """
    p = tl.load(p_dev)
    # Central region: p in (0.5, 1.0)
    q = p - 0.5
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as required by A&S
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


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, icdf, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification per row: ReLU(X - (mean + std * icdf)).
    OUT[b, s, f] = max(0, X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * icdf))
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_start = (b * S + s) * F
    row_ptr = X + row_start
    out_ptr = OUT + row_start

    mean = MEANS[pid]
    std = STDs[pid]
    threshold = mean + std * icdf

    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_ptr + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + idx, y, mask=mask)


# Entry point
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [batch_size, seq_len, intermediate_size] (bf16 or fp16/fp32)
        Computes adaptive sparsity threshold based on input statistics:
          threshold = mean + std * icdf(target_sparsity)
        Returns ReLU(x - threshold) in bfloat16 (same shape as x).
        """
        if target_sparsity == 0.0:
            return x

        # Ensure float32 compute; keep x contiguous
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, S, F = x.shape
        # Per-row means and stds
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Select BLOCK and meta-params based on F (constexpr choices)
        if F >= 4096:
            BLOCK_STATS = 4096
            num_warps_stats = 8
            num_stages_stats = 3
            BLOCK_SP = 4096
            num_warps_sp = 8
            num_stages_sp = 3
        elif F >= 2048:
            BLOCK_STATS = 2048
            num_warps_stats = 8
            num_stages_stats = 2
            BLOCK_SP = 2048
            num_warps_sp = 8
            num_stages_sp = 2
        else:
            BLOCK_STATS = 1024
            num_warps_stats = 4
            num_stages_stats = 2
            BLOCK_SP = 1024
            num_warps_sp = 4
            num_stages_sp = 2

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Allocate output
        out = torch.empty_like(x)

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)