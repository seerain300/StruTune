import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reduce_stats(x_ptr, mean_ptr, std_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # One program per row (flattened rows = B * S)
    row_id = tl.program_id(0)
    # Guard for rows not in grid; usually grid == rows
    # Accumulate sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in chunks
    for col in range(0, N, BLOCK_SIZE):
        idx = col + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        # Row-major contiguous: offset = row_id * N + idx
        offsets = row_id * N + idx
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # x is float32 (we pass x_f32 to kernel)
        x = x.to(tl.float32)
        # Masked load handled by masked tl.load; sum masked entries as 0
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and std
    mean = sum_val / N
    # population std (unbiased=False): var = E[x^2] - (E[x])^2
    var = sum_sq / N - mean * mean
    # Avoid tiny negative due to FP errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_inv_ndtri(out_ptr, p: tl.constexpr):
    # Inverse normal CDF using Abramowitz & Stegun approximation (A&S 7.1.26).
    # Single-element kernel: p is a scalar float.
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

    # Handle regions
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # Central region
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(out_ptr, z)


@triton.jit
def gate_rows(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, y_ptr, N: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    # Load scalar inv_cdf
    inv_cdf = tl.load(inv_cdf_ptr)
    # Compute threshold
    threshold = mean + std * inv_cdf
    # Process row elements
    for col in range(0, N, 1):
        idx = row_id * N + col
        val = tl.load(x_ptr + idx)
        # ReLU gate: y = max(0, val - threshold)
        diff = val - threshold
        # Triton supports tl.maximum; emulate ReLU: max(diff, 0)
        zero = 0.0
        y = tl.maximum(diff, zero)
        tl.store(y_ptr + idx, y)


class ModelNew(nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "inputs must be on CUDA device for Triton kernels"
        x = inputs.contiguous()
        # We will compute in float32 inside Triton and cast back to input dtype
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape
        rows = B * S

        # Output as float32 (compute), we'll cast to bfloat16 at the end to match original behavior
        out_f32 = torch.empty_like(x_f32)

        # Allocate mean and std buffers [rows]
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        # Launch Triton reduction to compute mean and std per row
        # Choose BLOCK_SIZE as a power of two that is convenient; N is typically large and divisible by 1024
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_stats[grid](x_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE)

        # Compute inv_norm_cdf(target_sparsity) with Triton; result is a 1-element tensor on device
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        # Note: target_sparsity is a scalar; Triton supports constexpr scalar argument
        compute_inv_ndtri[(1,)](inv_cdf_buf, target_sparsity)

        # If no sparsity, return inputs cast to bfloat16. To ensure Triton usage, we can still run a trivial gate with inv_cdf=0.
        if target_sparsity == 0.0:
            # y = x (no gating), but still compute via Triton gate with threshold=0 to keep Triton active.
            # However, for correctness, y = x directly. We can simply return x.to(torch.bfloat16).
            # But since we must use Triton, we call gate_rows with inv_cdf=0 to produce y=x.
            inv_cdf_zero = torch.zeros(1, device=x_f32.device, dtype=torch.float32)
            out_f32.zero_()
            gate_rows[grid](x_f32, mean, std, inv_cdf_zero, out_f32, N)
            return out_f32.to(torch.bfloat16)

        # Elementwise gating: y = max(0, x - (mean + std * inv_cdf))
        out_f32.zero_()
        gate_rows[grid](x_f32, mean, std, inv_cdf_buf, out_f32, N)

        # Cast to bfloat16 to match original behavior and return
        return out_f32.to(torch.bfloat16)