import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F, and stores sumsq[row] for later use.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over the feature dimension in tiles
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + offs
        mask = cols < F
        base = row * F + cols
        x = tl.load(X_ptr + base, mask=mask, other=0.0)
        # Reduce tile to scalars
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    # Write per-row statistics
    mean = total_sum / F
    sumsq = total_sumsq  # we will compute var/std on the host using unbiased=False
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, sumsq)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) of scalar p using A&S 7.1.26 approximation.
    Reads p from p_in_ptr (1-element tensor), writes to p_out_ptr.
    Clamp p to [eps, 1 - eps] to avoid log(0)/log(1).
    """
    p = tl.load(p_in_ptr)  # scalar
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants
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

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    t = c1 * q + c2
    t = t * q + c3
    t = t * q + c4
    t = t * q + c5
    t = t * q + c6
    num = t
    denom = d1 * q + d2
    denom = denom * q + d3
    denom = denom * q + d4
    denom = denom * q + 1.0
    ndtri_low = num / denom

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    num2 = a1 * r + a2
    num2 = num2 * r + a3
    num2 = num2 * r + a4
    num2 = num2 * r + a5
    num2 = num2 * r + a6
    den2 = b1 * r + b2
    den2 = den2 * r + b3
    den2 = den2 * r + b4
    den2 = den2 * r + b5
    ndtri_mid = num2 / den2

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    t2 = c1 * q3 + c2
    t2 = t2 * q3 + c3
    t2 = t2 * q3 + c4
    t2 = t2 * q3 + c5
    t2 = t2 * q3 + c6
    num3 = t2
    denom2 = d1 * q3 + d2
    denom2 = denom2 * q3 + d3
    denom2 = denom2 * q3 + d4
    denom2 = denom2 * q3 + 1.0
    ndtri_high = -num3 / denom2

    # Select based on p
    res = tl.zeros((), dtype=tl.float32)
    res = tl.where(p < p_low, ndtri_low, res)
    res = tl.where(p >= p_low, tl.where(p <= p_high, ndtri_mid, ndtri_high), res)
    tl.store(p_out_ptr, res)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, x - (mean + std * multiplier)) for each element.
    mean_ptr and std_ptr are per-row vectors of length ROWS. multiplier_ptr is a 1-element tensor.
    """
    row = tl.program_id(0)
    if row >= ROWS:
        return

    cutoff = tl.load(mean_ptr + row) + tl.load(std_ptr + row) * tl.load(multiplier_ptr)
    base = row * F
    offs = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, F, BLOCK_SIZE):
        cols = col_start + offs
        mask = cols < F
        x = tl.load(X_ptr + base + cols, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation using Triton kernels.

        Inputs:
            inputs: Tensor of shape [batch_size, seq_len, intermediate_size], dtype float32 or bfloat16, on CUDA.
            target_sparsity: float in (0, 1), target fraction of active elements.

        Returns:
            Sparsified tensor of same shape as input, dtype bfloat16.
        """
        assert inputs.is_cuda, "ModelNew requires CUDA tensors."
        B, S, F = inputs.shape
        rows = B * S

        # Work in float32 for stability; keep original dtype only for output
        x = inputs
        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        x32 = x.to(torch.float32)

        # 1) Compute per-row mean and sumsq in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32.view(rows, F),
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std = sqrt(sumsq/F - mean^2) on device using torch for efficiency
        var = sumsq / F - mean * mean
        std = torch.sqrt(var)

        # 3) Compute scalar multiplier via Triton ndtri approximation
        p_in = x32.new_tensor(target_sparsity)  # 1-element device scalar
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation via Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
