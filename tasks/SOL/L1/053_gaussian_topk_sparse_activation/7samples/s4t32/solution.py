import torch
import triton
import triton.language as tl


@triton.jit
def mean_sumsq_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    out_mean_ptr,         # *float32, output per (b, s): mean
    out_sumsq_ptr,        # *float32, output per (b, s): sumsq/F
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides for x
    BLOCK_F: tl.constexpr,           # chunk size along F
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for (b, s, 0)
    base = b * stride_b + s * stride_s

    # Accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Loop over feature dimension in chunks of BLOCK_F
    f = 0
    while f < F:
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    sumsq_mean = acc_sumsq / F
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, sumsq_mean)


@triton.jit
def invnorm_as_kernel(out_ptr, target_sparsity: tl.float32):
    # Compute invnorm(target_sparsity) using Abramowitz & Stegun 5.2.23 approximation.
    # invnorm(p) = sqrt(2) * erfinv(2p - 1). We approximate erfinv via solving erf(z) = 2p - 1 using AS7.1.26.
    # Constants (double precision math is fine; Triton uses fp32)
    p = target_sparsity  # provided as float32

    # Initial approximation for z
    p2 = 2.0 * p - 1.0
    # For p <= 0.5: use the lower region approximation
    # For p > 0.5: use the upper region approximation; we'll mirror logic.
    # Start with a robust guess z ~ sqrt(2*(p2 + tiny)) to avoid division by zero
    tiny = 1e-7
    z = tl.sqrt(2.0 * (p2 + tiny))

    # Newton iterations for erf(z) = p2
    # erf(z) ~ sign * [1 - t * exp(-z^2) * (a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4)], t = 1/(1+p z)
    # For our case, p2 in (-1, 1), and z >= 0, so we can use the positive branch directly.
    # We perform a few iterations to refine z.
    for _ in range(4):
        t = 1.0 / (1.0 + 0.3275911 * z)
        # Polynomial approximation (positive branch)
        # poly = (a1*t + a2)*(t + a3)*(t + a4)*(t + a5) + (a6*t^2 + a7*t^3 + a8*t^4)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        a6 = 0.3678309859
        a7 = 0.531656391
        a8 = -0.0240333019

        poly = ((a1 * t + a2) * (t + a3) * (t + a4) * (t + a5)) + \
               (a6 * t * t + a7 * t * t * t + a8 * t * t * t * t)
        erf_z = 1.0 - poly * tl.exp(-(z * z))
        # Update z: z_{n+1} = z_n + (p2 - erf(z_n)) / (2*sqrt(pi))
        # 2/sqrt(pi) = 1.1283791670955126
        z = z + (p2 - erf_z) * 1.1283791670955126

    # Store the refined z
    tl.store(out_ptr, z)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    mean_ptr,             # *float32, per (b, s) mean
    sumsq_ptr,            # *float32, per (b, s) sumsq/F (i.e., mean^2 + var)
    invnorm_ptr,          # *float32, scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output tensor
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input/output strides for x/out
    BLOCK_F: tl.constexpr,
):
    # 3D grid: (B, S, cdiv(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    start = chunk * BLOCK_F
    offs = b * stride_b + s * stride_s + start + tl.arange(0, BLOCK_F)
    mask = (start + tl.arange(0, BLOCK_F)) < F

    # Load a chunk of x
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load mean and std for (b, s)
    mean = tl.load(mean_ptr + b * S + s)
    sumsq = tl.load(sumsq_ptr + b * S + s)
    std = tl.sqrt(sumsq)

    invnorm = tl.load(invnorm_ptr)  # scalar

    threshold = mean + std * invnorm
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation:
    - Computes per-(batch, seq) mean and std across feature dim.
    - Computes invnorm(target_sparsity) in Triton.
    - Applies ReLU(x - (mean + std * invnorm)) elementwise in Triton.
    Returns tensor in bfloat16.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for computations
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and sumsq/F (std via sqrt(sumsq - mean^2) would require mean^2; we store sumsq/F directly)
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/std reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_sumsq_kernel[grid](
        x, mean, sumsq, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) in Triton (scalar)
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_as_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x, dtype=torch.float32)

    # Launch elementwise ReLU-threshold kernel: 3D grid over (B, S, F chunks)
    grid3 = (B, S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid3](
        x, mean, sumsq, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16
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
            # If more than two args, try to extract sparsity as the second arg
            if len(args) > 1 and isinstance(args[1], (float, int)):
                return run(args[0], float(args[1]))
            # Fallback
            return run(args[0], 0.01)


def run(*args):
    return ModelNew()(*args)
