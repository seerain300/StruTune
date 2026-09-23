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
    row_offset = b * S + s
    base = row_offset * F

    sum_x = 0.0
    sum_x2 = 0.0

    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        # X is laid out as contiguous [B*S, F] when we pass a [B*S, F] view; each row has length F
        x = tl.load(X + base + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / F
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_x2 / F - mean * mean
    # Ensure numerical stability: clamp variance to non-negative
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write out
    tl.store(MEANS + row_offset, mean)
    tl.store(STDs + row_offset, std)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for a given probability P using Abramowitz & Stegun 5.2.23
    central region approximation. P: [1] device tensor, ICDF: [1] device tensor (float32).
    """
    # Load probability
    p = tl.load(P)
    # Central region: p in [0.5, 1]
    q = p - 0.5
    r = q * q

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
    icdf_val = poly * q / den

    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = max(0, X - (MEANS + STDs * ICDF)) per element.
    MEANS, STDs: [B*S], broadcast along features.
    X, OUT: [B*S, F], row-major layout: row base = row * F.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row = b * S + s
    base_x = row * F
    base_out = row * F

    cutoff = tl.load(MEANS + row) + tl.load(STDs + row) * tl.load(ICDF)
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X + base_x + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(OUT + base_out + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        inputs: [batch_size, seq_len, intermediate_size]
        target_sparsity: float in [0, 1]
        Returns: bfloat16 tensor with adaptive sparsity threshold based on rowwise mean and std.
        """
        # Ensure dtype float32 for compute
        x = inputs.to(torch.float32)
        B, S, F = x.shape

        # Prepare output and per-row stats buffers
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Choose BLOCK and num_warps based on F
        if F >= 4096:
            BLOCK_STATS = BLOCK_SP = 4096
            num_warps_stats = 8
            num_warps_sp = 8
            num_stages_stats = 3
            num_stages_sp = 3
        elif F >= 2048:
            BLOCK_STATS = BLOCK_SP = 2048
            num_warps_stats = 8
            num_warps_sp = 8
            num_stages_stats = 2
            num_stages_sp = 2
        else:
            BLOCK_STATS = BLOCK_SP = 1024
            num_warps_stats = 4
            num_warps_sp = 4
            num_stages_stats = 2
            num_stages_sp = 2

        # Launch rowwise stats kernel: one program per (batch, seq) row
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

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)