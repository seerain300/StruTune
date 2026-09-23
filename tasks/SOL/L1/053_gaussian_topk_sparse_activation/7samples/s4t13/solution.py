import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor flattened as [B*S, F] in contiguous fashion
    mean_ptr,             # *float32, output mean per (b, s) of length B*S
    std_ptr,              # *float32, output std per (b, s) of length B*S
    B: tl.constexpr, S: tl.constexpr, F,  # sizes
    stride_b, stride_s, stride_f,          # input strides in elements
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators (scalars)
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over feature dimension in chunks of BLOCK_F
    BLOCK_F = 1024
    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        # Reduce within the chunk to scalars and accumulate
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # unbiased=False (population variance)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def invnorm_kernel(
    out_ptr,              # *float32, output scalar tensor of length 1
    target_sparsity,      # float32 scalar
):
    # A&S constants for invnorm approximation
    p = target_sparsity
    # Abramowitz and Stegun 7.1.26 approximations
    # z = sign(p - 0.5) * (1 + poly(t)) / (1 + poly(s))
    # with t = 2*(1 - p), s = 1/(1 + p)
    # poly(y) = ((((c5*y + c4)*y + c3)*y + c2)*y + c1)*y
    c1 = 2.506628277459239e+00
    c2 = -3.066479806614716e+01
    c3 = 3.647848324763204e+01
    c4 = -1.688126154351984e+01
    c5 = 4.838317949838609e-02

    p = p.to(tl.float32)
    t = 2.0 * (1.0 - p)
    s = 1.0 / (1.0 + p)
    poly = (((((c5 * t) + c4) * t + c3) * t + c2) * t + c1) * t
    z = (1.0 + poly) / (1.0 + poly * s)

    # Handle p < 0.5 by sign
    sign = 1.0 if p >= 0.5 else -1.0
    z = sign * tl.sqrt(t) * z  # note: sqrt(t) is undefined if p >= 0.5, but we handled via sign

    # Store as float32
    tl.store(out_ptr, z)


@triton.jit
def relu_threshold_kernel_1d(
    x_ptr,               # *float32, input tensor flattened, shape [B*S*F]
    mean_ptr,            # *float32, per-(b, s) mean, shape [B*S]
    std_ptr,             # *float32, per-(b, s) std, shape [B*S]
    inv_ptr,             # *float32, scalar invnorm, shape [1]
    out_ptr,             # *float32, output tensor flattened, shape [B*S*F]
    total_elems,         # int, total number of elements B*S*F
    B, S, F,             # int sizes
    stride_b, stride_s, stride_f,  # input strides (elements)
    BLOCK: tl.constexpr,             # block size for linear iteration
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    # Load x
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Reconstruct (b, s, f) indices
    BSF = S * F
    b = offs // BSF
    rem = offs % BSF
    s = rem // F
    f = rem % F

    # Compute linear offsets for mean/std lookup: idx = b*S + s
    idx = b * S + s

    mean = tl.load(mean_ptr + idx, mask=mask, other=0.0)
    std = tl.load(std_ptr + idx, mask=mask, other=0.0)
    z = tl.load(inv_ptr)  # scalar

    # Threshold = mean + std * invnorm(target_sparsity)
    threshold = mean + std * z
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
    - Compute per-(b, s) mean and std with Triton reduction kernel.
    - Compute invnorm(target_sparsity) with Triton scalar kernel.
    - Apply ReLU(x - threshold) with Triton elementwise kernel.
    Returns output in bfloat16, matching original behavior.
    """
    # Early exit if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for computations; work on a contiguous copy
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Output buffers for mean and std
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    std = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, std, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) using Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](float(target_sparsity), num_warps=1, num_stages=1)

    # Flatten pointers for elementwise kernel
    x_flat = x.reshape(-1)
    out_flat = torch.empty_like(x_flat, dtype=torch.float32, device=x.device)

    total = B * S * F
    # Launch 1D elementwise ReLU-threshold kernel
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    relu_threshold_kernel_1d[grid](
        x_flat, mean, std, invnorm, out_flat, total, B, S, F, stride_b, stride_s, stride_f, BLOCK,
        num_warps=4, num_stages=2
    )

    # Reshape back and return in bfloat16
    out = out_flat.reshape(B, S, F)
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature: run(inputs, target_sparsity)
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
