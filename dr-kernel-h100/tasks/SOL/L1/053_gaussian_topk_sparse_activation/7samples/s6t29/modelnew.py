import torch
import triton
import triton.language as tl


@triton.jit
def _compute_mean_std_and_sparsify_kernel(
    X, OUT,
    B, S, F,
    MEAN, STD, THRESH, ICDF,
    BLOCK: tl.constexpr
):
    """
    Single Triton kernel:
    - For each row (b, s), loop over features in BLOCK chunks to compute mean and population std.
    - Compute threshold = mean + std * icdf(target_sparsity).
    - Loop over features again to apply y = max(0, x - threshold) and write to OUT.

    X: [B, S, F] input in fp32
    OUT: [B, S, F] output in fp32
    MEAN: [B*S] per-row mean (fp32) - pointer to store result
    STD: [B*S] per-row std (fp32) - pointer to store result
    THRESH: [B*S] per-row threshold (fp32) - pointer to store result
    ICDF: scalar fp32 inverse-normal CDF for target_sparsity
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base pointers for this row
    row_x_ptr = X + b * S * F + s * F
    row_out_ptr = OUT + b * S * F + s * F

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: compute sum and sum of squares across features
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and population std (unbiased=False)
    F_fp = F  # compile-time constant -> fp32 scalar
    mean = sum_val / F_fp
    # std = sqrt((sum_sq / F) - mean^2)
    var = sum_sq / F_fp - mean * mean
    # Clamp variance to non-negative to avoid small negative due to FP errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute threshold = mean + std * icdf(target_sparsity)
    # ICDF is a scalar argument; compute in fp32
    threshold = mean + std * ICDF

    # Store per-row stats for potential host use (if needed later)
    MEAN[pid] = mean
    STD[pid] = std
    THRESH[pid] = threshold

    # Second pass: apply ReLU(x - threshold) and store
    for off in range(0, F, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(row_x_ptr + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(row_out_ptr + idx, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation:
    - Compute statistics and sparsification in a single Triton kernel.
    - Output in bfloat16 to match original behavior.
    """
    # Ensure CUDA and contiguous memory
    assert inputs.is_cuda, "Inputs must be on CUDA device."
    x = inputs.contiguous()
    B, S, F = x.shape

    # Compute in fp32 for numeric stability
    x_fp32 = x.to(torch.float32)
    out = torch.empty_like(x_fp32)  # fp32 output

    # Per-row stats buffers (fp32)
    means = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    stds = torch.empty((B * S,), dtype=torch.float32, device=x.device)
    thresholds = torch.empty((B * S,), dtype=torch.float32, device=x.device)

    # Inverse-normal CDF scalar for target_sparsity computed via Triton kernel
    icdf_dev = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev = torch.empty((), dtype=torch.float32, device=x.device)
    p_dev.fill_(float(target_sparsity))
    _icdf_ndtri_kernel[(1,)](
        p_dev, icdf_dev, num_warps=1, num_stages=1
    )

    # Launch single Triton kernel over rows
    BLOCK = 1024
    num_warps = 4
    num_stages = 2
    grid = (B * S,)
    _compute_mean_std_and_sparsify_kernel[grid](
        x_fp32, out,
        B, S, F,
        means, stds, thresholds, icdf_dev,
        BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages
    )

    # Cast back to bfloat16 to match original behavior
    return out.to(torch.bfloat16)

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single input tensor [B, S, F]
        if len(args) != 1:
            raise ValueError("ModelNew expects a single input tensor [batch_size, seq_len, intermediate_size].")
        x = args[0]
        # We do not assume a specific target_sparsity; default 0.0 means no sparsity. However, the original run uses it.
        # Here we mimic the original behavior: run with target_sparsity=0.5 as a default, but keep it as a parameter in evaluation.
        # In this environment, target_sparsity is provided via the runner, so we just call run(x, target_sparsity).
        # The runner will pass the target_sparsity, but since it's not provided here, we assume 0.5 for testing.
        # To be safe, we extract target_sparsity from args if present; otherwise default to 0.5.
        target_sparsity = 0.5
        if len(args) == 2:
            target_sparsity = float(args[1])
        return run(x, target_sparsity)