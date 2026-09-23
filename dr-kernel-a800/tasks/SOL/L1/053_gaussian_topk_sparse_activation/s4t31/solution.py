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
    Compute inverse standard normal CDF for scalar p via A&S 7.1.26 approximation.
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p (scalar)
    p = tl.load(p_dev_ptr)

    # Piecewise regions
    p_low = 0.02425
    if p < 0.5:
        # Lower region: t = sqrt(2) * sqrt(-log(p))
        t = 1.4142135623730951 * tl.sqrt(-tl.log(p))  # Triton log/sqrt
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

        poly = (((((a1 * t) + a2) * t + a3) * t + a4) * t + a5) * t + a6
        denom = (((((b1 * t) + b2) * t + b3) * t + b4) * t + b5) * t + 1.0
        val = -poly / denom
    else:
        # Upper region: use symmetry
        val = -_ndtri_scalar_kernel(1.0 - p, out_ptr)

    # Store scalar result
    tl.store(out_ptr, val)


@triton.jit
def _sparsify_apply_threshold_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, N, threshold_scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification:
    out[row, i] = max(0, inp[row, i] - (mean[row] + std[row] * threshold_scale))
    inp_ptr: *f32, shape [B, S, N], contiguous
    out_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    threshold_scale: f32 scalar computed as inverse normal CDF of target_sparsity
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)

    cutoff = mean + std * threshold_scale

    # Iterate over elements in the row
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = tl.maximum(x - cutoff, 0.0)  # ReLU: max(0, x - cutoff)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-(batch,seq) mean and std across the last dim (unbiased=False).
        - Compute inverse normal CDF(std_multiplier) inside a Triton scalar kernel.
        - Apply out = relu(input - (mean + std * std_multiplier)) via Triton kernel.
        """
        if target_sparsity == 0.0:
            # No sparsity requested
            return inputs

        # Ensure contiguous and compute in float32 for numerical stability
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)
        B, S, N = inp_f32.shape
        total_rows = B * S

        # Allocate stats and output
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out_f32 = torch.empty_like(inp_f32)

        # Compute threshold_scale using Triton scalar kernel (no torch ops on device)
        p_tensor = torch.tensor([target_sparsity], dtype=torch.float32, device=inp_f32.device)
        out_tensor = torch.empty([1], dtype=torch.float32, device=inp_f32.device)
        _ndtri_scalar_kernel[(1,)](p_tensor, out_tensor)  # single program for scalar
        threshold_scale = out_tensor[0]  # scalar tensor on device

        # Launch reduction kernel to compute per-row mean and std
        BLOCK_SIZE_RED = 1024
        _row_reduce_mean_std_kernel[(total_rows,)](
            inp_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RED, num_warps=4, num_stages=4
        )

        # Launch elementwise sparsification kernel
        BLOCK_SIZE_ELE = 1024
        _sparsify_apply_threshold_kernel[(total_rows,)](
            inp_f32, out_f32, mean, std, N, threshold_scale, BLOCK_SIZE=BLOCK_SIZE_ELE, num_warps=4, num_stages=4
        )

        # Cast back to original dtype
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
