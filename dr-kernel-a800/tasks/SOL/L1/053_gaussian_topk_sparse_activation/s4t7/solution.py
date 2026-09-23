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

    # Iterate over row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        # Accumulate scalars
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _row_sparsify_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, N, std_multiplier: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, apply thresholding: out = relu(inp - (mean + std * std_multiplier)).
    inp_ptr: *f32, input row base
    out_ptr: *f32, output row base
    mean_ptr: *f32, per-row mean
    std_ptr: *f32, per-row std
    N: int, length of last dim
    std_multiplier: Python float scalar (host-computed inverse normal CDF)
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row stats
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)

    # Compute cutoff
    cutoff = mean + std * std_multiplier

    # Process the row
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU: max(y, 0)
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run():
        - Compute per-(batch, seq) mean and std across last dim.
        - Compute adaptive cutoff threshold: mean + std * inv_norm(target_sparsity).
        - Apply ReLU(input - threshold) elementwise.
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous().to(torch.float32)
        B, S, N = inp.shape

        # Buffers for mean and std
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inp.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inp.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B * S,)
        BLOCK_SIZE = 2048  # larger block to reduce loop iterations
        _row_reduce_mean_std_kernel[grid](inp, mean_buf, std_buf, N, BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute std_multiplier on host using original Python helper (no torch ops on device tensors here)
        # This helper is pure Python and returns a Python float; no device tensor is created in forward.
        std_multiplier_scalar = float(_ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inp.device)))

        # Output buffer (float32 for compute)
        out = torch.empty_like(inp)

        # Launch sparsification kernel
        _row_sparsify_kernel[grid](inp, out, mean_buf, std_buf, N, std_multiplier_scalar, BLOCK_SIZE, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out.to(inputs.dtype)


# Original helper: inverse standard normal CDF using A&S 7.1.26 approximation
def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    Works well for p in (0, 1).
    """
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

    result = torch.zeros_like(p)

    # Lower region
    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid.any():
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    mask_high = p > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
        result[mask_high] = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return result


# Example usage remains the same: ModelNew takes inputs and target_sparsity, returns sparsified tensor.


def run(*args):
    return ModelNew()(*args)
