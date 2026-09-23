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
    # Map program id to (b, s)
    b = pid // S
    s = pid % S

    # Accumulators in fp32
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over features in BLOCK-sized chunks
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        # Compute linear index for X[b, s, idx]
        x_ptrs = X + b * S * F + s * F + idx
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and population std (unbiased=False)
    F_f = tl.float32(F)
    mean = sum_x / F_f
    var = sum_x2 / F_f - mean * mean
    # std = sqrt(max(var, 0)) to avoid tiny negative due to FP error
    std = tl.sqrt(tl.maximum(var, 0.0))

    # Write results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT[b, s, f] = relu(X[b, s, f] - (MEANS[pid] + STDs[pid] * ICDF))
    One program per (b, s) row. ICDF is a scalar computed from target_sparsity.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    cutoff = mean + std * ICDF

    # Iterate over features in BLOCK-sized chunks, apply relu(x - cutoff)
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x_ptrs = X + b * S * F + s * F + idx
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        y = tl.maximum(x_vals - cutoff, 0.0)
        out_ptrs = OUT + b * S * F + s * F + idx
        tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a single probability P[0] using Abramowitz & Stegun 5.2.23 central region.
    Store result in ICDF[0].
    """
    # Load probability
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q

    # Coefficients (central region)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # must be negative
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly_num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    num = poly_num * q
    icdf = num / poly_den

    # Store result
    tl.store(ICDF, icdf)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        1) Compute per-row mean and std across last dimension (features).
        2) Compute icdf(target_sparsity) using Triton scalar kernel.
        3) Apply sparsification: y = relu(x - (mean + std * icdf)) per element.
        Returns in bfloat16.
        """
        # Ensure contiguous
        x = x.contiguous()
        B, S, F = x.shape

        # Compute in fp32 for numerical stability
        x_fp32 = x.to(torch.float32)

        # Allocate outputs and per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Fixed meta-parameters for Triton (robust across shapes)
        BLOCK_STATS = 1024
        NUM_WARPS_STATS = 4
        BLOCK_SP = 1024
        NUM_WARPS_SP = 4

        # Launch rowwise stats kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=NUM_WARPS_STATS, num_stages=2
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
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=NUM_WARPS_SP, num_stages=2
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)