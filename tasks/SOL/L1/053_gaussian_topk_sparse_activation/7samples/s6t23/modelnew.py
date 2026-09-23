import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_reduce_sum_sumsq_kernel(X, B, S, F, SUMS, SUMSQS, BLOCK: tl.constexpr):
    """
    One program per (b, s) row. Accumulate sum and sum of squares across features F.
    X: [B, S, F] input (float32)
    SUMS: [B*S] (float32), SUMSQS: [B*S] (float32)
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_ptr = X + b * S * F + s * F
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over features in chunks of BLOCK with masks for tail
    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        # Masked accumulation: ensure masked elements don't contribute
        x = tl.where(mask, x, 0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Write results for this row
    out_index = pid  # linearized (b, s) index
    tl.store(SUMS + out_index, sum_val)
    tl.store(SUMSQS + out_index, sumsq_val)


@triton.jit
def _rowwise_sparsify_relu_kernel(X, SUMS, SUMSQS, icdf, OUT, B, S, F, BLOCK: tl.constexpr):
    """
    One program per (b, s) row. Use precomputed sums and sumsqs to compute mean and std,
    then output = relu(x - (mean + std * icdf)).
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    n = F  # number of features per row
    sum_val = tl.load(SUMS + pid)
    sumsq_val = tl.load(SUMSQS + pid)
    mean = sum_val / n
    # population std (unbiased=False)
    var = sumsq_val / n - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    row_ptr = X + b * S * F + s * F
    out_ptr = OUT + b * S * F + s * F

    # Compute threshold per row: mean + std * icdf
    threshold = mean + std * icdf

    # Apply sparsification: output = relu(x - threshold)
    for start in range(0, F, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < F
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        x = tl.where(mask, x, 0.0)
        y = x - threshold  # scalar threshold subtracted elementwise
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _icdf_ndtri_kernel(p_dev, icdf, BLOCK: tl.constexpr):
    """
    Compute inverse normal CDF for a single probability p_dev into icdf using
    Abramowitz & Stegun 5.2.23 central region formula. This is a scalar kernel.
    """
    # Load probability
    p = tl.load(p_dev)
    # Central region approximation
    # Constants for A&S 5.2.23 central region
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01  # note: negative a5
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # q = p - 0.5
    q = p - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly / den
    # Store result
    tl.store(icdf, z)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation using Triton.
        Inputs: x of shape [batch_size, seq_len, intermediate_size]
        Returns: sparsified tensor of same shape as input, in bfloat16.
        """
        # Ensure contiguous and compute in fp32
        x = x.contiguous()
        x_fp32 = x.to(torch.float32)
        B, S, F = x_fp32.shape

        # Allocate per-row sums and sumsqs
        sums = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        sumsq = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Fixed compile-time constants for Triton kernels to ensure stability
        BLOCK = 1024
        NUM_WARPS = 4
        NUM_STAGES = 2

        # Launch reduction kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_reduce_sum_sumsq_kernel[grid](
            x_fp32, B, S, F, sums, sumsq, BLOCK=BLOCK, num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )

        # Compute icdf(target_sparsity) via Triton scalar kernel
        icdf = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_dev.fill_(float(target_sparsity))
        _icdf_ndtri_kernel[(1,)](
            p_dev, icdf, BLOCK=1, num_warps=1, num_stages=1
        )

        # Output buffer in fp32
        out = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Launch sparsify + ReLU kernel
        _rowwise_sparsify_relu_kernel[grid](
            x_fp32, sums, sumsq, icdf, out, B, S, F, BLOCK=BLOCK, num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)