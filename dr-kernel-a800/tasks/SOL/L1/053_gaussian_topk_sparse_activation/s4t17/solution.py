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
    num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        off = chunk * BLOCK_SIZE
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
    Compute inverse standard normal CDF for scalar p via Abramowitz & Stegun 7.1.26.
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p
    p = tl.load(p_dev_ptr)

    # Constants for the approximation
    # Lower region constants
    p_low = 0.02425
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

    # Central region constants (same as lower region constants above)
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

    # Compute piecewise inverse normal CDF
    if p < 0.5:
        # Lower region: p in (0, 0.5)
        # x = sqrt(-2*log(p)) -> for p near 0, x large. Triton has log/sqrt; use them directly.
        x = tl.sqrt(-2.0 * tl.log(p))
        # Rational approximation
        num = (((((c1 * x + c2) * x + c3) * x + c4) * x + c5) * x + c6)
        den = ((((d1 * x + d2) * x + d3) * x + d4) * x + 1.0)
        z = -((num / den))
        tl.store(out_ptr, z)
    else:
        # Central region: p in [0.5, 1)
        # t = p - 0.5
        t = p - 0.5
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        t6 = t5 * t

        P = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5 + a6 * t6
        Q = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5

        z = t * (P / Q)
        tl.store(out_ptr, z)


@triton.jit
def _sparsify_row_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, std_multiplier_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise kernel: out = relu(inp - (mean + std * std_multiplier)) for each row.
    inp_ptr: *f32, shape [B, S, N]
    out_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    std_multiplier_ptr: *f32, shape [1] (scalar tensor)
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    std_multiplier = tl.load(std_multiplier_ptr)  # scalar

    cutoff = mean + std * std_multiplier

    # Iterate over the row in chunks and apply thresholding + ReLU
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
          - If target_sparsity == 0.0: return inputs.
          - Else: compute per-row mean and std across last dim via Triton,
                  compute inv-std-normal multiplier via Triton (A&S 7.1.26),
                  then apply adaptive threshold ReLU via Triton.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous float32 for computation
        inputs = inputs.contiguous()
        inputs_f32 = inputs.to(torch.float32)

        B, S, N = inputs_f32.shape

        # Allocate per-row stats (mean and std)
        mean = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel to compute mean and std
        BLOCK_SIZE = 2048
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inputs_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute inv-std-normal multiplier using Triton (scalar)
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        std_multiplier = torch.empty([1], dtype=torch.float32, device=inputs.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier)

        # Allocate output
        out = torch.empty_like(inputs_f32)

        # Launch sparsification kernel
        _sparsify_row_kernel[grid](inputs_f32, out, mean, std, std_multiplier, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
