import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    # Map pid to (b, s)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s) row
    base = (b * S + s) * D

    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Loop over feature dimension in tiles
    for off in range(0, D, 1024):
        idx = off + tl.arange(0, 1024)
        mask = idx < D
        # Load a tile and accumulate
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    # Store per-row sums
    tl.store(SUM_ptr + pid, acc_sum)
    tl.store(SUMSQ_ptr + pid, acc_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D                # int32
):
    pid = tl.program_id(axis=0)
    sum_ = tl.load(SUM_ptr + pid)
    sumsq_ = tl.load(SUMSQ_ptr + pid)

    d_f = tl.full((), D, tl.float32)
    mean = sum_ / d_f
    var = sumsq_ / d_f - mean * mean
    # Guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF.
    # We'll implement for p in (0, 1). The host will pass a single scalar.

    # Load p (assume OUT_ptr[0] holds the scalar p)
    p = tl.load(OUT_ptr + 0)
    # Constants
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

    # Output z
    z = tl.zeros((), dtype=tl.float32)

    # Lower region
    mask_low = p < p_low
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / denom

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly * q / denom

    # Upper region
    mask_high = p > p_high
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = -poly / denom

    # Store result
    tl.store(OUT_ptr + 0, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    ZSCORE_ptr,      # *float32, length 1
    OUT_ptr,         # *float32, length B*S*D (we'll store float32 then cast)
    B, S, D          # int32
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    # Load mean and std for this (b, s)
    mean = tl.load(MEAN_ptr + (b * S + s))
    std = tl.load(STD_ptr + (b * S + s))
    z_score = tl.load(ZSCORE_ptr + 0)

    threshold = mean + std * z_score

    base = (b * S + s) * D
    off = tile * 1024 + tl.arange(0, 1024)
    mask = off < D

    x = tl.load(X_ptr + base + off, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = tl.maximum(x - threshold, 0.0)

    # Store as float32 to OUT_ptr (host will cast to bfloat16)
    tl.store(OUT_ptr + base + off, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure device and contiguity
        assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, D = inputs.shape

        # Output buffer as float32 for activation kernel; we'll cast to bfloat16 after.
        out_fp32 = torch.empty(B * S * D, device=inputs.device, dtype=torch.float32)

        # Buffers for per-row sums and sumsq
        sums = torch.empty(B * S, device=inputs.device, dtype=torch.float32)
        sumsq = torch.empty(B * S, device=inputs.device, dtype=torch.float32)
        mean = torch.empty(B * S, device=inputs.device, dtype=torch.float32)
        std = torch.empty(B * S, device=inputs.device, dtype=torch.float32)

        # Scalar z-score buffer (1 element)
        zscore_buf = torch.empty(1, device=inputs.device, dtype=torch.float32)
        # Write the sparsity into zscore_buf[0] so ndtri_approx_kernel has the scalar
        zscore_buf[0] = float(target_sparsity)

        # Launch reduction
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](inputs, sums, sumsq, B, S, D, num_warps=8, num_stages=2)

        # Launch mean/std
        compute_mean_std_kernel[grid_reduce](sums, sumsq, mean, std, D, num_warps=1, num_stages=1)

        # Launch ndtri approximation to compute z_score
        ndtri_approx_kernel[(1,)](zscore_buf, num_warps=1, num_stages=1)

        # Launch apply activation
        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](inputs, mean, std, zscore_buf, out_fp32, B, S, D, num_warps=8, num_stages=2)

        # Cast to bfloat16 to match original output dtype
        return out_fp32.view(B, S, D).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
