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

    # Accumulators in f32
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
def _sparsify_row_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, apply adaptive threshold: out = relu(inp - (mean + std * std_multiplier)).
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N], contiguous
    N: int, length of last dim
    std_multiplier: f32 scalar
    """
    pid = tl.program_id(0)
    row_start = pid * N
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * std_multiplier

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function) using A&S approximation (7.1.26)."""
    # Constants for the approximation
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

    # Central region
    q = p - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_r = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly * q / poly_r

    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Upper region
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return z


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        - Compute per-row mean and std over last dimension.
        - Compute adaptive cutoff threshold = mean + std * norm.icdf(target_sparsity).
        - Apply ReLU(input - threshold) to create sparse activations.
        """
        # Handle no sparsity
        if target_sparsity == 0.0:
            # Return a fresh tensor with same dtype and device
            return inputs.clone()

        # Ensure contiguous and compute in float32
        inputs_f32 = inputs.to(torch.float32).contiguous()
        B, S, N = inputs_f32.shape
        device = inputs_f32.device

        # Allocate per-row stats
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch reduction kernel
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inputs_f32, mean, std, N, BLOCK_SIZE=2048, num_warps=8, num_stages=4)

        # Compute std multiplier using A&S approximation on host (no torch ops on device tensors in forward)
        std_multiplier = _ndtri(target_sparsity)

        # Allocate output
        out_f32 = torch.empty_like(inputs_f32)

        # Launch elementwise sparsification kernel
        _sparsify_row_kernel[grid](inputs_f32, mean, std, out_f32, N, std_multiplier, BLOCK_SIZE=4096, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
