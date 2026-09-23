import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B*S, F] input (float32), flattened rows
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    F_row = F
    # Each program processes one row of length F
    offs = tl.arange(0, BLOCK)
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in chunks of BLOCK with mask for tail
    for start in range(0, F_row, BLOCK):
        idx = start + offs
        mask = idx < F_row
        # X is a [B*S, F] contiguous row; pointer offset for row pid
        x = tl.load(X + pid * F_row + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and population std (unbiased=False)
    Ff = F_row  # int, Triton will handle division; use float via Python scope
    mean = sum_val / Ff
    var = sum_sq / Ff - mean * mean
    std = tl.sqrt(var)

    # Store results
    tl.store(MEANS + pid, mean)
    tl.store(STDs + pid, std)


@triton.jit
def _icdf_ndtri_kernel(P, OUT, BLOCK: tl.constexpr):
    # Compute icdf(P) for scalar P using Abramowitz & Stegun 5.2.23 central region formula.
    # OUT is a 1-element tensor (same device, float32).
    # Load scalar probability
    p = tl.load(P)  # shape: ()
    # Central region: q = p - 0.5
    q = p - 0.5
    r = q * q
    # Polynomial coefficients (float32)
    # a1..a6, b1..b5 as provided in the original code
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # negative as per Abramowitz & Stegun
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly * q / den

    # Store as scalar
    tl.store(OUT, icdf)


@triton.jit
def _sparsify_relu_kernel(X_ROWS, MEANS, STDs, ICDF, OUT_ROWS, F, BLOCK: tl.constexpr):
    """
    Apply elementwise: OUT[i] = max(0, X_ROWS[i] - (MEANS[i] + STDs[i] * ICDF))
    X_ROWS: [B*S, F] input (float32)
    MEANS: [B*S] (float32)
    STDs: [B*S] (float32)
    ICDF: 1-element tensor (float32)
    OUT_ROWS: [B*S, F] output (float32)
    """
    pid = tl.program_id(axis=0)
    F_row = F
    offs = tl.arange(0, BLOCK)
    mean = tl.load(MEANS + pid)
    std = tl.load(STDs + pid)
    cutoff = mean + std * tl.load(ICDF)  # scalar
    for start in range(0, F_row, BLOCK):
        idx = start + offs
        mask = idx < F_row
        x = tl.load(X_ROWS + pid * F_row + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ROWS + pid * F_row + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-based implementation of Gaussian-based top-k sparse activation:
        - Compute per-(batch, seq) mean and std along last dim.
        - Compute icdf(target_sparsity) using A&S 5.2.23 central region formula.
        - Apply ReLU(input - (mean + std * icdf)) to create sparse activations.
        Returns bfloat16 tensor of same shape as input.
        """
        if target_sparsity == 0.0:
            return x

        device = x.device
        dtype = torch.float32

        # Work in fp32 for numeric stability
        x_fp32 = x.to(dtype)

        B, S, F = x_fp32.shape
        # Flatten to [B*S, F] per-row for simple indexing
        x_rows = x_fp32.view(B * S, F)

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=dtype, device=device)
        stds = torch.empty((B * S,), dtype=dtype, device=device)

        # Choose BLOCK and launch params based on F (compile-time constants)
        if F >= 2048:
            BLOCK_STATS = 2048
            BLOCK_SP = 2048
            num_warps_stats = 8
            num_warps_sp = 8
            num_stages_stats = 3
            num_stages_sp = 3
        else:
            BLOCK_STATS = 1024
            BLOCK_SP = 1024
            num_warps_stats = 4
            num_warps_sp = 4
            num_stages_stats = 2
            num_stages_sp = 2

        # Launch rowwise stats kernel
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_rows, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=dtype, device=device)
        p_dev = torch.empty((), dtype=dtype, device=device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Prepare output per-row
        out_rows = torch.empty((B * S, F), dtype=dtype, device=device)

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_rows, means, stds, icdf, out_rows, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Reshape back to [B, S, F] and return in bfloat16
        out = out_rows.view(B, S, F).to(torch.bfloat16)
        return out