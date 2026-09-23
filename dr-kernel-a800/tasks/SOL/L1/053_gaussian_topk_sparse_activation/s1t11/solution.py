import torch
import triton
import triton.language as tl


@triton.jit
def fused_reduce_and_activate_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous, float32
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D], contiguous
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    base = b * S + s
    base_idx = base * D  # scalar int32 base offset

    # First pass: compute sum and sum of squares across D
    sum_val = 0.0
    sumsq_val = 0.0

    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        ptrs = X_ptr + base_idx + idx
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    # Compute mean and std
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)  # clamp tiny negatives
    std = tl.sqrt(var)

    # Load z_score (scalar) and compute threshold
    z_score = tl.load(Z_ptr)  # float32 scalar
    threshold = mean + std * z_score

    # Second pass: apply activation y = max(0, x - threshold)
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        ptrs = X_ptr + base_idx + idx
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        out_ptrs = OUT_ptr + base_idx + idx
        tl.store(out_ptrs, y, mask=mask)
        offs += BLOCK_SIZE


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar sparsity p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p
    p = tl.load(P_ptr)
    # Constants for A&S 5.2.23 approximation (Abramowitz & Stegun)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    z_low = poly / den

    # Central region
    q2 = p - 0.5
    r2 = q2 * q2
    poly_mid = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6)
    den_mid = (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)
    z_mid = poly_mid * q2 / den_mid

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_hi = (((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6)
    den_hi = ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)
    z_hi = -poly_hi / den_hi

    # Select region based on p
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    z = z_mid
    z = tl.where(cond_low, z_low, z)
    z = tl.where(cond_mid, z_mid, z)
    z = tl.where(~(cond_low | cond_mid), z_hi, z)

    tl.store(OUT_ptr, z)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        - All computation is done by Triton kernels.
        - No torch elementwise ops or reductions in forward (host code).
        """
        assert inputs.dim() == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        B, S, D = inputs.shape

        # Ensure input is contiguous and compute in float32 for stability
        X = inputs.contiguous()
        if X.dtype != torch.float32:
            X = X.to(torch.float32)

        # Buffer for z_score (scalar) computed by Triton


def run(*args):
    return ModelNew()(*args)
