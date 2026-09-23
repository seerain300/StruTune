import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(
    x_ptr,                # *float32, input tensor as float32
    out_mean_ptr,         # *float32, output mean per (b, s)
    out_var_ptr,          # *float32, output variance per (b, s)
    B, S, F,              # int sizes
    stride_b, stride_s, stride_f,  # input strides
    BLOCK_F: tl.constexpr
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Compute base pointer for this (b, s)
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Iterate over feature dimension F in chunks of BLOCK_F
    for f in range(0, F, BLOCK_F):
        offs = base + f + tl.arange(0, BLOCK_F)
        mask = (f + tl.arange(0, BLOCK_F)) < F
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    mean = acc_sum / F
    var = acc_sumsq / F - mean * mean  # population variance
    # Store per-(b, s) mean and variance
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_var_ptr + pid, var)


@triton.jit
def relu_threshold_kernel(
    x_ptr,         # *float32, input tensor
    mean_ptr,      # *float32, per-(b, s) mean
    var_ptr,       # *float32, per-(b, s) variance
    invnorm,       # *float32 scalar tensor (size 1)
    out_ptr,       # *float32, output tensor
    B, S, F,       # sizes
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr
):
    # 2D grid: (B*S, cdiv(F, BLOCK_F))
    pid0 = tl.program_id(0)  # over (b, s)
    pid1 = tl.program_id(1)  # over feature chunks
    b = pid0 // S
    s = pid0 % S

    base = b * stride_b + s * stride_s
    f = pid1 * BLOCK_F

    offs = base + f + tl.arange(0, BLOCK_F)
    mask = (f + tl.arange(0, BLOCK_F)) < F

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Load per-(b, s) stats
    mean = tl.load(mean_ptr + pid0)
    var = tl.load(var_ptr + pid0)
    std = tl.sqrt(var)

    # invnorm scalar
    z = tl.load(invnorm)

    # Apply ReLU(x - (mean + std * z))
    threshold = mean + std * z
    y = x - threshold
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version:
      1) Compute per-(b, s) mean and variance across the last dim.
      2) Compute invnorm(target_sparsity) in a Triton scalar kernel.
      3) Apply ReLU(x - (mean + std * invnorm)) in a Triton elementwise kernel.
      4) Return output in bfloat16 to match original behavior.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        # Return input in bfloat16
        return inputs.to(torch.bfloat16)

    # Ensure float32 for numerical stability and contiguous layout
    x = inputs.to(torch.float32).contiguous()
    B, S, F = x.shape
    stride_b, stride_s, stride_f = x.stride()

    # Allocate outputs for mean and variance (per (b, s))
    mean = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    var = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch mean/var reduction kernel: one program per (b, s)
    grid = (B * S,)
    mean_std_kernel[grid](
        x, mean, var, B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Compute invnorm(target_sparsity) in a Triton scalar kernel
    invnorm = torch.empty((1,), dtype=torch.float32, device=x.device)
    invnorm_kernel[invnorm](target_sparsity, num_warps=1, num_stages=1)

    # Output buffer
    out = torch.empty_like(x)

    # Elementwise ReLU-threshold kernel: 2D grid over (B*S, F chunks)
    grid2 = (B * S, triton.cdiv(F, 1024))
    relu_threshold_kernel[grid2](
        x, mean, var, invnorm, out,
        B, S, F, stride_b, stride_s, stride_f,
        BLOCK_F=1024, num_warps=4, num_stages=2
    )

    # Return in bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Keep original signature expectation: inputs tensor, target_sparsity float
        # Delegate to run(inputs, target_sparsity) if provided.
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