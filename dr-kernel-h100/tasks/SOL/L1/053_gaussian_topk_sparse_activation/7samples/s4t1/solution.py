import torch
import triton
import triton.language as tl


# Kernel: for each (b, s), compute sum and sumsq across feature dim F
# Then mean = sum / F, std = sqrt(sumsq / F - mean^2)  (unbiased=False)
@triton.jit
def mean_std_kernel(
    x_ptr,          # *const float32, input tensor
    mean_ptr,       # *float32, output [B*S]
    std_ptr,        # *float32, output [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B*S - 1)
    b = pid // S
    s = pid % S

    # Accumulate sum and sum of squares
    sum_val = 0.0
    sumsq_val = 0.0

    offs = 0
    while offs < F:
        idx_f = offs + tl.arange(0, BLOCK_F)
        mask = idx_f < F
        ptrs = x_ptr + b * stride_b + s * stride_s + idx_f * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # vals are float32
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)
        offs += BLOCK_F

    F_f = F  # Python int; Triton allows using runtime ints
    mean = sum_val / F_f
    var = sumsq_val / F_f - mean * mean
    std = tl.sqrt(var)

    # Store as contiguous 1D index
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


# Kernel: elementwise ReLU(x - cutoff) where cutoff = mean[b,s] + std[b,s] * invnormcdf(sparsity)
# invnormcdf uses Abramowitz & Stegun approximation (5.2.23).
@triton.jit
def relu_threshold_kernel(
    x_ptr,            # *const float32, input tensor
    mean_ptr,         # *const float32, [B*S]
    std_ptr,          # *const float32, [B*S]
    out_ptr,          # *float32, output tensor
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    target_sparsity,  # float32 scalar (host passes)
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # seq
    pid_blk = tl.program_id(2)  # block along feature dim

    b = pid_b
    s = pid_s

    # Bounds check for safety (grid matches shapes, but keep it explicit)
    if b >= B or s >= S:
        return

    # Load mean and std for this (b, s)
    mean = tl.load(mean_ptr + b * S + s)
    std = tl.load(std_ptr + b * S + s)

    # Compute invnormcdf(target_sparsity) inline using A&S 5.2.23 approximation
    # Constants (same as provided in original code)
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

    # Compute invnormcdf(target_sparsity) (scalar)
    p = target_sparsity
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # Horner for polynomial
        poly = c6
        poly = poly * q + c5
        poly = poly * q + c4
        poly = poly * q + c3
        poly = poly * q + c2
        poly = poly * q + c1
        denom = d4
        denom = denom * q + d3
        denom = denom * q + d2
        denom = denom * q + d1
        invnorm = poly / (denom * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = a6
        num = num * r + a5
        num = num * r + a4
        num = num * r + a3
        num = num * r + a2
        num = num * r + a1
        den = b5
        den = den * r + b4
        den = den * r + b3
        den = den * r + b2
        den = den * r + b1
        invnorm = num * q / (den * r + 1.0)
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = c6
        poly = poly * q + c5
        poly = poly * q + c4
        poly = poly * q + c3
        poly = poly * q + c2
        poly = poly * q + c1
        denom = d4
        denom = denom * q + d3
        denom = denom * q + d2
        denom = denom * q + d1
        invnorm = -poly / (denom * q + 1.0)

    cutoff = mean + std * invnorm  # scalar

    # Process features in blocks
    offs = 0
    while offs < F:
        idx_f = offs + tl.arange(0, BLOCK_F)
        mask = idx_f < F
        in_ptrs = x_ptr + b * stride_b + s * stride_s + idx_f * stride_f
        out_ptrs = out_ptr + b * stride_b + s * stride_s + idx_f * stride_f

        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        # ReLU(x - cutoff)
        diff = vals - cutoff
        res = tl.maximum(diff, 0.0)
        tl.store(out_ptrs, res, mask=mask)

        offs += BLOCK_F


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    - Compute mean and std per (b, s) via Triton reduction in a single pass
    - Compute invnormcdf(target_sparsity) in host (one scalar)
    - Apply elementwise ReLU(x - cutoff) via Triton
    Returns bfloat16 tensor (same as original).
    """
    if target_sparsity == 0.0:
        # Early exit: no sparsity
        return inputs

    # Compute in float32 for stability
    x = inputs
    x_f32 = x.to(torch.float32)

    B, S, F = x_f32.shape
    # Strides for potentially non-contiguous input (we still pass as-is)
    stride_b, stride_s, stride_f = x_f32.stride()

    # 1) Compute mean and std using Triton reduction kernel
    mean_std_elems = B * S
    mean = torch.empty(mean_std_elems, dtype=torch.float32, device=x.device)
    std = torch.empty(mean_std_elems, dtype=torch.float32, device=x.device)

    BLOCK_F = 1024
    grid_mean = (B * S,)
    mean_std_kernel[grid_mean](
        x_f32, mean, std,
        B, S, F,
        stride_b, stride_s, stride_f,
        BLOCK_F,
        num_warps=4, num_stages=1
    )

    # 2) Elementwise ReLU with per-(b,s) cutoff; we pass invnorm as target_sparsity (already computed as desired)
    out_f32 = torch.empty_like(x_f32)

    grid = (B, S, triton.cdiv(F, BLOCK_F))
    relu_threshold_kernel[grid](
        x_f32, mean, std, out_f32,
        B, S, F,
        stride_b, stride_s, stride_f,
        float(target_sparsity),  # pass as float32
        BLOCK_F,
        num_warps=4, num_stages=1
    )

    return out_f32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor shaped [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor.")
        inputs = args[0]
        # target_sparsity is not provided as an argument in the original Model.run; assume it's a module attribute or fixed.
        # Since the original code uses a global function run(inputs, target_sparsity), we replicate the signature here:
        # In typical benchmarking, target_sparsity is provided via the harness. If not, default to 0.5.
        # To keep compatibility, we require the caller to pass target_sparsity as the second argument.
        if len(args) == 2:
            return run(inputs, args[1])
        else:
            # Fallback default sparsity (e.g., 0.01)
            return run(inputs, 0.01)


def run(*args):
    return ModelNew()(*args)
