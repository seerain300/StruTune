import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor (float32)
    out_mean_ptr,         # *float32, per-(b, s) mean
    out_sumsq_ptr,        # *float32, per-(b, s) sum of squares
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension in chunks
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, 1024)
        mask = (f + tl.arange(0, 1024)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += 1024

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, acc_sumsq / F)


@triton.jit
def invnorm_kernel(  # Computes invnorm(target_sparsity) using A&S approximation
    out_ptr,             # *float32, single-element output
    target_sparsity,     # float32 scalar
):
    # A&S constants
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

    # piecewise regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # since we only have one scalar, compute unified invnorm
    p = target_sparsity

    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        r = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        d = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        inv = r / d
    else:
        # Upper region
        if p > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            r = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            d = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
            inv = -r / d
        else:
            # Central region
            q = p - 0.5
            r = q * q
            r2 = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            r3 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            inv = r2 * q / r3

    tl.store(out_ptr, inv)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor
    mean_ptr,             # *float32, per-(b, s) mean
    sumsq_ptr,            # *float32, per-(b, s) sum of squares / F
    invnorm_ptr,          # *float32, scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # sizes
    stride_b, stride_s, stride_f,  # input/output strides
):
    # 3D grid: (B, S, chunks along F)
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    base = b * stride_b + s * stride_s
    start = chunk * 1024
    offs = base + start + tl.arange(0, 1024)
    mask = (start + tl.arange(0, 1024)) < F

    # Load x chunk
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load per-(b, s) mean and std
    pid = b * S + s
    mean = tl.load(mean_ptr + pid)
    sumsq = tl.load(sumsq_ptr + pid)
    std = tl.sqrt(sumsq)  # population std

    # Load scalar invnorm(target_sparsity)
    inv = tl.load(invnorm_ptr)

    # Compute threshold and ReLU
    threshold = mean + std * inv
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation of:
      cutoff per (b, s): mean + std * invnorm(target_sparsity)
      output = max(0, inputs - cutoff)
    Works for input shape [batch_size, seq_len, intermediate_size].
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 and contiguous for Triton
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate per-(b, s) mean and sum of squares
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Allocate scalar invnorm and launch Triton kernel to compute it
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](float(target_sparsity), num_warps=1, num_stages=1)

    # Output tensor
    out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature: run(inputs, target_sparsity)
        if len(args) == 2:
            return run(args[0], float(args[1]))
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            return run(args[0], 0.01)
        else:
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)


def run(*args):
    return ModelNew()(*args)
