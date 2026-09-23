import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_sum_sumsq_kernel(inp_ptr, sum_ptr, sumsq_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute sum and sum of squares across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
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

    tl.store(sum_ptr + pid, sum_x)
    tl.store(sumsq_ptr + pid, sum_x2)


@triton.jit
def _compute_mean_std_from_sums_kernel(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, N):
    """
    Compute mean and std for each row from sum and sumsq.
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int
    """
    pid = tl.program_id(0)
    s = tl.load(sum_ptr + pid)
    ss = tl.load(sumsq_ptr + pid)
    mean = s / N
    var = ss / N - mean * mean
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _sparsify_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(input - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, last dimension size
    std_multiplier: f32 scalar
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * std_multiplier

    # Iterate over the row in chunks and apply ReLU(input - threshold)
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        z = x - threshold
        z = tl.maximum(z, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, z, mask=mask)


def _ndtri(p: float) -> float:
    """
    Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 26.2.23), with precise constants.
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

    result = 0.0

    # Lower region
    if p < p_low:
        q = torch.sqrt(-2.0 * torch.log(p))  # this log will be called by the host; not Triton
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return result

    # Central region
    if p <= p_high:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return result

    # Upper region
    q = torch.sqrt(-2.0 * torch.log(1.0 - p))  # host-side, no Triton
    result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    return result


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation:
        Computes adaptive sparsity threshold per (batch, seq) row:
          threshold = mean(input_row) + std(input_row) * inv_stdnormal(target_sparsity)
        Then applies out = relu(input - threshold).

        Triton is used for statistics and elementwise sparsification.
        The inverse-normal scalar is computed on the host using the original precise formula.
        """
        # Early exit: no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure dtype float32 for compute; keep original dtype for return
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)

        B, S, N = inp_f32.shape
        total_rows = B * S

        # Allocate accumulators and stats
        sum_vec = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        sumsq_vec = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out_f32 = torch.empty_like(inp_f32)

        # Launch reduction kernel: compute per-row sum and sumsq
        BLOCK_SIZE_REDUCE = 1024 if N >= 1024 else 256
        _row_reduce_sum_sumsq_kernel[(total_rows,)](
            inp_f32, sum_vec, sumsq_vec, N, BLOCK_SIZE=BLOCK_SIZE_REDUCE, num_warps=4, num_stages=3
        )

        # Compute mean and std from sums on host (no Triton op)
        mean_cpu = sum_vec / float(N)
        std_cpu = torch.sqrt(sumsq_vec / float(N) - mean_cpu * mean_cpu)
        mean = mean_cpu.to(device=inp_f32.device)
        std = std_cpu.to(device=inp_f32.device)

        # Compute std_multiplier (inverse standard normal CDF of target_sparsity) on host using precise formula
        std_multiplier = float(_ndtri(float(target_sparsity)))

        # Launch elementwise sparsification kernel
        BLOCK_SIZE_ELEM = 1024 if N >= 1024 else 256
        _sparsify_kernel[(total_rows,)](
            inp_f32, mean, std, out_f32, N, std_multiplier, BLOCK_SIZE=BLOCK_SIZE_ELEM, num_warps=4, num_stages=3
        )

        # Cast back to original dtype and return
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
