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

    # Base index into flattened [B*S, F]
    base = b * S + s

    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over features in chunks of BLOCK
    for start in range(0, F, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        # Pointer to the start of this row in X
        ptr = X + base * F + idx
        x = tl.load(ptr, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # population std (unbiased=False), var could be slightly negative due to FP rounding; clamp to >= 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    MEANS[base] = mean
    STDs[base] = std


@triton.jit
def _icdf_ndtri_kernel(P, ICDF, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF (quantile) for given probability P (shape [1]) using A&S 5.2.23 central region.
    Writes result to ICDF (shape [1]).
    """
    # Load probability
    p = tl.load(P)
    # Central region: p in [0.5, 1]
    q = p - 0.5
    r = q * q
    poly = (
        (((((2.506628277459239e+00 * r + (-3.066479806614716e+01)) * r + 1.383577518672690e+02) * r
          + (-2.759285104469687e+02)) * r + 2.209460984245205e+02) * r + (-3.969683028665376e+01)) * r
        + 1.0
    )
    # Note: a5 is negative as per Abramowitz & Stegun 5.2.23
    denom = (
        (((((2.938163982698783e+00 * r + 4.374664141464968e+00) * r + (-2.549732539343734e+00)) * r
          + (-2.400758277161838e+00)) * r + (-3.223964580411365e-01)) * r + (-7.784894002430293e-03)) * r
        + 1.0
    )
    icdf_val = poly * q / denom
    tl.store(ICDF, icdf_val)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT[b, s, f] = max(0, X[b, s, f] - (MEANS[b*S + s] + STDs[b*S + s] * ICDF))
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S + s
    mean = MEANS[base]
    std = STDs[base]
    cutoff = mean + std * tl.load(ICDF)

    for start in range(0, F, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < F
        in_ptr = X + base * F + idx
        x = tl.load(in_ptr, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        out_ptr = OUT + base * F + idx
        tl.store(out_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run:
        - Compute per-row mean and std across last dimension (feature) in fp32.
        - Compute inverse-normal CDF for target_sparsity as a scalar in Triton.
        - Apply sparsification: max(0, x - (mean + std * icdf)), return in bfloat16.
        """
        # Ensure CUDA tensors
        if not x.is_cuda:
            raise RuntimeError("ModelNew.forward requires a CUDA tensor input.")

        # Compute in fp32 for numerical stability
        x_fp32 = x.to(torch.float32)

        B, S, F = x_fp32.shape

        # Allocate outputs for per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Launch rowwise stats kernel with fixed BLOCK to ensure robustness
        BLOCK_STATS = 1024
        num_warps_stats = 4
        num_stages_stats = 2
        _rowwise_stats_kernel[(B * S,)](
            x_fp32, B, S, F, means, stds, BLOCK=BLOCK_STATS, num_warps=num_warps_stats, num_stages=num_stages_stats
        )

        # Compute icdf for target_sparsity using Triton (single scalar)
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Allocate output buffer
        out = torch.empty_like(x_fp32)

        # Launch sparsify + ReLU kernel
        BLOCK_SP = 1024
        num_warps_sp = 4
        num_stages_sp = 2
        _sparsify_relu_kernel[(B * S,)](
            x_fp32, means, stds, icdf, out, B, S, F, BLOCK=BLOCK_SP, num_warps=num_warps_sp, num_stages=num_stages_sp
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
