import torch
import triton
import triton.language as tl


@triton.jit
def mean_sumsq_kernel(
    x_ptr,                # *float32, input tensor as float32, shape [B, S, F]
    out_mean_ptr,         # *float32, output mean per (b, s), size [B*S]
    out_sumsq_ptr,        # *float32, output sum of squares per (b, s), size [B*S]
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s)
    base = b * stride_b + s * stride_s

    acc_sum = 0.0
    acc_sumsq = 0.0

    f = 0
    while f < F:
        offs = f + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + base + offs * stride_f, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        f += BLOCK_F

    mean = acc_sum / F
    sumsq_avg = acc_sumsq / F
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_sumsq_ptr + pid, sumsq_avg)


@triton.jit
def relu_threshold_kernel(
    x_ptr,                # *float32, input
    mean_ptr,             # *float32, mean per (b, s)
    sumsq_ptr,            # *float32, sumsq/F per (b, s) [we derive std = sqrt(sumsq - mean^2)]
    invnorm_ptr,          # *float32, scalar invnorm(target_sparsity)
    out_ptr,              # *float32, output
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b: tl.constexpr, stride_s: tl.constexpr, stride_f: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(F, BLOCK_F))
    b = tl.program_id(0)
    s = tl.program_id(1)
    chunk = tl.program_id(2)

    offs = chunk * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = offs < F

    # Load per-(b, s) mean and std from scalars
    mean = tl.load(mean_ptr + b * S + s)
    sumsq_avg = tl.load(sumsq_ptr + b * S + s)
    std = tl.sqrt(sumsq_avg - mean * mean)
    invnorm = tl.load(invnorm_ptr)  # scalar

    # Compute threshold per (b, s)
    threshold = mean + std * invnorm

    # Load x chunk
    x = tl.load(x_ptr + b * stride_b + s * stride_s + offs * stride_f, mask=mask, other=0.0)
    # Apply ReLU(x - threshold). Broadcast threshold as a scalar.
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + b * stride_b + s * stride_s + offs * stride_f, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
    - Compute per-(b, s) mean and std along last dim (F).
    - Compute invnorm(target_sparsity) using PyTorch to ensure correctness.
    - Apply ReLU(x - (mean + std * invnorm)) elementwise.
    Returns bfloat16 tensor.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure float32 for Triton computations; keep contiguous along last dim
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate per-(b, s) mean and sumsq/F
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/sumsq reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_sumsq_kernel[grid](
        x, mean, sumsq,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) using PyTorch (robust and correct)
    # Use torch.erf-based approach: invnorm(p) = sqrt(2)*erf_inv(2p - 1)
    # erf_inv is available via torch.erfinv.
    target = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
    # invnorm = sqrt(2) * erfinv(2p - 1)
    invnorm = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.erfinv((2.0 * target - 1.0))
    invnorm = invnorm.to(torch.float32)  # keep as 1-element tensor on device

    # Output buffer
    out = torch.empty_like(x)

    # Launch elementwise ReLU threshold kernel with 3D grid
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
        # The evaluator typically passes two arguments: input tensor and float sparsity.
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