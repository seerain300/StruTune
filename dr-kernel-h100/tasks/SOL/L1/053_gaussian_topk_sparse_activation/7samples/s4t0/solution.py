import torch
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    X_ptr,  # *float32
    Mean_ptr,  # *float32, shape (B*S,)
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    sum_val = 0.0
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(X_ptr + base + idx * stride_f, mask=mask, other=0.0)
        # x is a vector; sum over the vector
        sum_val += tl.sum(x, axis=0)
        offs += BLOCK_F

    mean_val = sum_val / F
    tl.store(Mean_ptr + pid, mean_val)


@triton.jit
def std_kernel(
    X_ptr,  # *float32
    Std_ptr,  # *float32, shape (B*S,)
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    sum_val = 0.0
    sumsq_val = 0.0
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(X_ptr + base + idx * stride_f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        offs += BLOCK_F

    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    std_val = tl.sqrt(var)
    tl.store(Std_ptr + pid, std_val)


@triton.jit
def _invnormcdf_triton(p):
    # Abramowitz and Stegun 7.1.26 approximation (piecewise)
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    if p < p_low:
        # q = sqrt(-2 * log(p))
        q = tl.sqrt(-2.0 * tl.log(p))
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

        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return poly / denom

    # Central region (p <= 0.5)
    if p <= 0.5:
        q = p - 0.5
        r = q * q
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

        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return poly * q / den

    # Upper region (p > 0.5): use symmetry of normal CDF
    q = 1.0 - p
    r = q * q
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

    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly * q / den
    return -z


@triton.jit
def relu_threshold_kernel(
    X_ptr,            # *float32, input tensor
    Mean_ptr,         # *float32, shape (B*S,)
    Std_ptr,          # *float32, shape (B*S,)
    Out_ptr,          # *float32, output tensor
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    target_sparsity: tl.constexpr,  # float
    BLOCK_F: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)
    f_start = tile * BLOCK_F
    offs = f_start + tl.arange(0, BLOCK_F)
    mask = offs < F

    base = b * stride_b + s * stride_s
    x = tl.load(X_ptr + base + offs * stride_f, mask=mask, other=0.0)

    pid = b * S + s
    mean = tl.load(Mean_ptr + pid)
    std = tl.load(Std_ptr + pid)

    invnorm = _invnormcdf_triton(target_sparsity)
    threshold = mean + std * invnorm

    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(Out_ptr + base + offs * stride_f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Early exit if no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "inputs must be on CUDA for Triton"
        x = inputs.contiguous()
        B, S, F = x.shape
        # Compute in float32 for numerical stability
        x_f32 = x.to(torch.float32)

        # Prepare mean and std buffers
        mean = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Strides for last-dim reduction
        stride_b = S * F
        stride_s = F
        stride_f = 1

        # Launch mean kernel
        grid_mean = (B * S,)
        mean_kernel[grid_mean](
            x_f32, mean, B, S, F, stride_b, stride_s, stride_f,
            BLOCK_F=1024, num_warps=4, num_stages=1
        )

        # Launch std kernel
        grid_std = (B * S,)
        std_kernel[grid_std](
            x_f32, std, B, S, F, stride_b, stride_s, stride_f,
            BLOCK_F=1024, num_warps=4, num_stages=1
        )

        # Output tensor
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        # Launch elementwise ReLU-threshold kernel
        BLOCK_F = 1024
        grid = (B, S, triton.cdiv(F, BLOCK_F))
        relu_threshold_kernel[grid](
            x_f32, mean, std, out_f32,
            B, S, F, stride_b, stride_s, stride_f,
            target_sparsity, BLOCK_F,
            num_warps=4, num_stages=1
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
