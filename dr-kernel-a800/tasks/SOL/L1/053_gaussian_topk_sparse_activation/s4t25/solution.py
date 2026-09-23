import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _sparsify_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification:
    out[i] = max(inp[i] - (mean[j] + std[j] * std_multiplier), 0)
    where j is the row index for element i (one row per program).
    inp_ptr: *f32, shape [B*S*N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B*S*N], contiguous
    N: int, last-dim size
    std_multiplier: f32 scalar (ndtri(target_sparsity))
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row stats
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * std_multiplier

    # Process the entire row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        # Apply adaptive threshold: max(x - threshold, 0)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


def _ndtri_scalar(p: float) -> float:
    """
    Host-side helper for scalar inverse standard normal CDF using Abramowitz & Stegun (7.1.26).
    Returns float.
    """
    # Constants for A&S approximation
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

    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return num / den
    elif p <= 0.5:
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return num / den
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        num = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        return num / den


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and std with Triton.
        - Compute std_multiplier = ndtri(target_sparsity) as a host scalar.
        - Apply adaptive threshold and ReLU via Triton elementwise kernel.
        Returns: same shape as inputs, dtype matches original (compute is done in f32 then cast back).
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in f32
        x = inputs.contiguous()
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)

        # Shapes
        B, S, N = x32.shape
        total = B * S
        device = x32.device

        # Allocate stats buffers
        mean = torch.empty(total, dtype=torch.float32, device=device)
        std = torch.empty(total, dtype=torch.float32, device=device)

        # Launch reduction kernel (tuned)
        BLOCK_SIZE_R = 2048  # good default; reduces loop iterations for N up to 16k
        grid = (total,)
        _row_reduce_mean_std_kernel[grid](x32, mean, std, N, BLOCK_SIZE_R, num_warps=8, num_stages=4)

        # Compute std_multiplier (inverse standard normal CDF) as a host scalar
        std_multiplier = _ndtri_scalar(float(target_sparsity))

        # Allocate output
        out32 = torch.empty_like(x32)

        # Launch sparsification kernel
        BLOCK_SIZE_E = 2048  # matches reduction block for consistent performance
        grid2 = (total,)
        _sparsify_kernel[grid2](x32, mean, std, out32, N, std_multiplier, BLOCK_SIZE_E, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out32.to(orig_dtype)


def run(*args):
    return ModelNew()(*args)
