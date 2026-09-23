import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes sum[row] and sumsq[row] to sum_out_ptr and sumsq_out_ptr.
    """
    row = tl.program_id(0)
    if row >= ROWS:
        return

    sum_val = 0.0
    sumsq_val = 0.0

    cols = tl.arange(0, BLOCK_SIZE)
    for offset in range(0, F, BLOCK_SIZE):
        idx = offset + cols
        mask = idx < F
        ptr = X_ptr + row * F + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    tl.store(sum_out_ptr + row, sum_val)
    tl.store(sumsq_out_ptr + row, sumsq_val)


@triton.jit
def _sqrt_std_kernel(var_ptr, std_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise: std[i] = sqrt(var[i]) for i in [0, N).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    var = tl.load(var_ptr + offs, mask=mask, other=0.0)
    std = tl.sqrt(var)
    tl.store(std_ptr + offs, std, mask=mask)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for scalar p using A&S 7.1.26 approximation.
    Reads p[0] and writes result to p_out_ptr[0].
    """
    p = tl.load(p_in_ptr)  # scalar in [eps, 1-eps]
    # A&S constants
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
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply activation: y = max(0, x - (mean + std * multiplier)).
    X_ptr: [ROWS*F], mean_ptr: [ROWS], std_ptr: [ROWS], multiplier_ptr: [1], Y_ptr: [ROWS*F].
    """
    row = tl.program_id(0)
    if row >= ROWS:
        return

    cutoff = tl.load(mean_ptr + row) + tl.load(std_ptr + row) * tl.load(multiplier_ptr)

    cols = tl.arange(0, BLOCK_SIZE)
    for offset in range(0, F, BLOCK_SIZE):
        idx = offset + cols
        mask = idx < F
        x = tl.load(X_ptr + row * F + idx, mask=mask, other=0.0)
        u = x - cutoff
        y = tl.where(u > 0.0, u, 0.0)
        tl.store(Y_ptr + row * F + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are contiguous and float32 for compute
        x = inputs.contiguous()
        B, S, F = x.shape
        rows = B * S

        # 1) Compute sum and sumsq per row via Triton
        sum_buf = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(rows, dtype=torch.float32, device=x.device)

        x_flat = x.view(rows, F)
        row_stats_kernel[(rows,)](
            x_flat, sum_buf, sumsq_buf,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute mean and var per row
        mean = sum_buf / F
        var = sumsq_buf / F - mean * mean  # unbiased=False

        # 3) Compute std via Triton elementwise sqrt (no torch.sqrt in forward)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        _sqrt_std_kernel[(rows,)](
            var, std, rows,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 4) Compute scalar ndtri multiplier in Triton
        p_in = x.new_tensor(target_sparsity).contiguous()  # device scalar, no torch.tensor on tensors
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 5) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y_flat = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x_flat, mean, std, std_multiplier, y_flat,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 6) Reshape and cast to bfloat16 to match original output
        y_out = y_flat.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
