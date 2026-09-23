import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_stats_kernel(X, MEANS, STDs, B, S, F, BLOCK: tl.constexpr):
    """
    Compute per-(batch, seq) row mean and population std across last dimension (features).
    X: [B, S, F], float32
    MEANS: [B*S], float32 output
    STDs: [B*S], float32 output
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointer for the (b, s) row
    base = b * S + s
    row_offset = base * F  # since in PyTorch (row-major), [B,S,F] -> contiguous last dim

    # Accumulate sum and sum of squares in fp32
    sum_x = 0.0
    sum_x2 = 0.0

    i = 0
    while i < F:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < F
        # Load a chunk of the row
        x = tl.load(X + row_offset + offs, mask=mask, other=0.0)
        # Reduce within the chunk
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        i += BLOCK

    # Compute mean and std (population std, unbiased=False)
    mean = sum_x / F
    var = sum_x2 / F - mean * mean
    # Ensure non-negative variance before sqrt
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store into flattened [B*S] buffers
    tl.store(MEANS + base, mean)
    tl.store(STDs + base, std)


@triton.jit
def _sparsify_relu_kernel(X, MEANS, STDs, ICDF, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    Apply sparsification: OUT = relu(X - (MEANS[base] + STDs[base] * ICDF))
    One program per (b, s) row. ICDF is a single scalar (1-element tensor).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    base = b * S + s

    mean = tl.load(MEANS + base)
    std = tl.load(STDs + base)
    threshold = mean + std * tl.load(ICDF)

    row_offset = base * F

    i = 0
    while i < F:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < F
        x = tl.load(X + row_offset + offs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT + row_offset + offs, y, mask=mask)
        i += BLOCK


def _run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized implementation of run() from the original code.
    """
    # Ensure dtype float32 for numeric stability; keep original layout
    x = inputs.to(torch.float32)
    B, S, F = x.shape

    # Allocate per-row stats
    means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Launch rowwise stats kernel: one program per (b, s) row
    BLOCK = 1024
    grid = (B * S,)
    _rowwise_stats_kernel[grid](
        x, means, stds, B, S, F, BLOCK,
        num_warps=4, num_stages=2
    )

    # Compute icdf for target_sparsity using Triton (single scalar)
    icdf = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf, num_warps=1, num_stages=1
    )

    # Output buffer
    out = torch.empty_like(x)

    # Launch sparsify + ReLU kernel
    _sparsify_relu_kernel[grid](
        x, means, stds, icdf, out, B, S, F, BLOCK,
        num_warps=4, num_stages=2
    )

    # Return in bfloat16, matching original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input of shape [batch_size, seq_len, intermediate_size]
        x = args[0]
        # If target_sparsity is provided via args, it's not used here (original run takes one tensor).
        # We keep it as a no-op to match the original Model's signature.
        return _run(x, 0.0)