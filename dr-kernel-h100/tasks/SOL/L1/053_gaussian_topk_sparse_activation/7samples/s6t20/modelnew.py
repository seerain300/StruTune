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

    acc = 0.0
    acc2 = 0.0

    offs = 0
    while offs < F:
        idx = row_start + offs + tl.arange(0, BLOCK)
        mask = idx < (row_start + F)
        x = tl.load(X + idx, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
        acc2 += tl.sum(x * x, axis=0)
        offs += BLOCK

    mean = acc / F
    ex2 = acc2 / F
    var = ex2 - mean * mean
    # population std (unbiased=False)
    std = tl.sqrt(var)
    MEANS[pid] = mean
    STDs[pid] = std


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    """
    Compute inverse standard normal CDF for a given probability P (scalar on device).
    Uses Abramowitz & Stegun 5.2.23 central region approximation:
    z = (P - 0.5) * (1 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)
    where t = 1 / (1 + p*|z|), p=0.3275911
    Note: a5 must be negative in A&S; here a5 = -3.066479806614716e+01.
    OUT: 1-element tensor to store icdf
    """
    # Load probability
    p = tl.load(P)
    # Constants
    p_const = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = -3.066479806614716  # negative per A&S

    # Work in float32
    p = p.to(tl.float32)
    # q = p - 0.5 (central region)
    q = p - 0.5
    # If q is too close to zero, avoid division by tiny
    # But here q is in (-0.5, 0.5); safe to compute t.
    z = q
    # t = 1 / (1 + p*|z|)
    z_abs = tl.abs(z)
    t = 1.0 / (1.0 + p_const * z_abs)
    # Evaluate polynomial
    # t0 = 1; we need t, t^2, ..., t^5
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t)  # evaluate at t (not t^5)
    z = q * (1.0 + poly)
    # Store result
    tl.store(OUT, z)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Elementwise sparsification:
    out[b,s,f] = relu( X[b,s,f] - (MEANS[b*S + s] + STDs[b*S + s] * ICDF) )
    MEANS: [B*S] float32
    STDs: [B*S] float32
    ICDF: scalar float32
    OUT: [B,S,F] float32
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_start = (b * S + s) * F

    thresh = MEANS[pid] + STDs[pid] * tl.load(ICDF)
    offs = 0
    while offs < F:
        idx = row_start + offs + tl.arange(0, BLOCK)
        mask = idx < (row_start + F)
        x = tl.load(X + idx, mask=mask, other=0.0)
        # ReLU: max(0, x - thresh)
        y = x - thresh
        y = tl.maximum(y, 0.0)
        tl.store(OUT + idx, y, mask=mask)
        offs += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Convert input to float32 for compute
        x = x.to(torch.float32)

        B, S, F = x.shape

        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Heuristic: use larger BLOCK when safe to reduce loop iterations
        if F >= 2048:
            BLOCK_STATS = 2048
            num_warps_stats = 8
            BLOCK_SP = 2048
            num_warps_sp = 8
        else:
            BLOCK_STATS = 1024
            num_warps_stats = 4
            BLOCK_SP = 1024
            num_warps_sp = 4

        # Launch rowwise stats kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)