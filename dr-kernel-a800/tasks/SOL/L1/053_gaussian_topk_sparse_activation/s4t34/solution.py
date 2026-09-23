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
def _compute_mean_std_from_sums_kernel(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute mean and std from per-row sum and sum of squares.
    mean = sum / N
    var = sumsq / N - mean^2  (population variance, unbiased=False)
    std = sqrt(var)
    """
    pid = tl.program_id(0)
    # Load per-row sum and sumsq
    s = tl.load(sum_ptr + pid)
    ss = tl.load(sumsq_ptr + pid)
    n = N
    mean = s / n
    var = ss / n - mean * mean
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S 7.1.26 approximation.
    Loads p from p_dev_ptr, stores result to out_ptr[0].
    """
    p = tl.load(p_dev_ptr)
    # A&S 7.1.26 constants
    p_low = 0.02425
    p_high = 1.0 - p_low
    use_low = p < p_low

    # Lower region approximation
    c1 = -7.784695709041462e-03
    c2 = 3.224671290700398e-01
    c3 = 2.445134137142996e+00
    c4 = 3.754408661907416e+00
    # Upper region approximation (negative branch)
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    if use_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + 1.0))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + 1.0)) / z
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = -(((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)) / z

    tl.store(out_ptr, result)


@triton.jit
def _sparsify_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification:
    out = relu(input - (mean + std * std_multiplier))
    Works across rows: pid indexes a single row, iterate over N in chunks.
    inp_ptr, out_ptr: *f32, shape [B, S, N] (linearized, contiguous)
    mean_ptr, std_ptr: *f32, shape [B*S]
    std_multiplier: scalar f32
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
        # relu(x - threshold)
        y = tl.maximum(x - threshold, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward: computes adaptive sparsity threshold based on input statistics:
        1) mean and std per (batch, seq) row along last dim (N)
        2) std_multiplier = inverse standard normal CDF of target_sparsity (A&S 7.1.26), computed in Triton
        3) out = relu(input - (mean + std * std_multiplier))
        Returns sparsified tensor with original dtype.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32 for numerical stability
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)
        B, S, N = inp_f32.shape
        total_rows = B * S

        # Allocate buffers
        sum_vec = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        sumsq_vec = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out_f32 = torch.empty_like(inp_f32)

        # 1) Per-row sum and sumsq via Triton
        BLOCK_SIZE_REDUCE = 1024 if N >= 1024 else 256
        _row_reduce_sum_sumsq_kernel[(total_rows,)](
            inp_f32, sum_vec, sumsq_vec, N, BLOCK_SIZE=BLOCK_SIZE_REDUCE, num_warps=4, num_stages=3
        )

        # 2) Compute mean and std from sums in Triton
        _compute_mean_std_from_sums_kernel[(total_rows,)](
            sum_vec, sumsq_vec, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_REDUCE, num_warps=4, num_stages=3
        )

        # 3) Compute std_multiplier (inverse normal CDF) in Triton using A&S 7.1.26
        p_dev = torch.tensor(float(target_sparsity), dtype=torch.float32, device=inp_f32.device)
        std_multiplier_buf = torch.empty((), dtype=torch.float32, device=inp_f32.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier_buf)

        # 4) Elementwise sparsification in Triton
        BLOCK_SIZE_ELEM = 1024 if N >= 1024 else 256
        _sparsify_kernel[(total_rows,)](
            inp_f32, mean, std, out_f32, N, std_multiplier_buf.item(), BLOCK_SIZE=BLOCK_SIZE_ELEM, num_warps=4, num_stages=3
        )

        # Cast back to original dtype
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
