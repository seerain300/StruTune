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
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p (scalar)
    p = tl.load(p_dev_ptr)

    # Constants for A&S approximation (7.1.26)
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

    # Initialize result
    result = 0.0

    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Store result
    tl.store(out_ptr, result)


@triton.jit
def _sparsify_kernel(inp_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification per row:
      cutoff = mean + std * z
      out[i] = relu(inp[i] - cutoff)
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    z_ptr: *f32, shape [1] (scalar inv-std-normal)
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        val = x - cutoff
        val = tl.maximum(val, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
            target_sparsity: Float in [0, 1], target sparsity level; 0.0 means no sparsity.

        Returns:
            Sparsified tensor of same shape as input (compute in float32, cast back).
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous()
        inp_f32


def run(*args):
    return ModelNew()(*args)
