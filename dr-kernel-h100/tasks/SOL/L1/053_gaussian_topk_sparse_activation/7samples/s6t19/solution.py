import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, B, S, F, MEANS, STDs, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F] input (float32), row-major
    MEANS: [B*S] output means (float32)
    STDs: [B*S] output stds (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer to this row
    row_base = X + b * S * F + s * F

    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over features in chunks of BLOCK
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # Clamp variance to non-negative for numerical stability (shouldn't be needed with fp32, but safe)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    means[pid] = mean
    stds[pid] = std


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply elementwise sparsity: OUT = relu(X - (MEANS + STDs * ICDF)),
    where MEANS/STDs are per-row scalars, ICDF is a scalar.
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_base = X + b * S * F + s * F
    out_row_base = OUT + b * S * F + s * F

    mean = MEANS[pid]
    std = STDs[pid]
    threshold = mean + std * ICDF

    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_base + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_row_base + idx, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for p using Abramowitz & Stegun 5.2.23 central region.
    P: 1-element tensor with target_sparsity (float32)
    ICDF: 1-element tensor to store result (float32)
    """
    # Read p
    p = tl.load(P)
    # Central region: q = p - 0.5
    q = p - 0.5

    # Coefficients (A&S 5.2.23)
    # a1..a5, b1..b5 from the original code (kept here for completeness)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # a5 negative as per A&S
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    icdf = poly / denom

    # Store result
    tl.store(ICDF, icdf)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [batch_size, seq_len, intermediate_size], any floating dtype
        target_sparsity: float in [0, 1]
        Returns: same shape as x, sparsified with threshold per row.
        """
        # Early return for no sparsity
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32
        x = x.contiguous()
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Allocate per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

        # Fixed meta-parameters for robustness
        BLOCK = 1024
        grid = (B * S,)

        # Launch rowwise stats kernel
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK, num_warps=4, num_stages=2
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Output buffer
        out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Launch sparsify + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=4, num_stages=2
        )

        # Cast back to original dtype (bfloat16 expected by the original code)
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
