import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F and sumsq[row] to mean_out_ptr and sumsq_out_ptr.
    """
    pid = tl.program_id(0)  # row id
    # Accumulators as scalars
    acc_sum = 0.0
    acc_sumsq = 0.0
    # Loop over features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Linear index into the flattened [ROWS, F] layout: idx = row * F + col
        idx = pid * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
    # Compute mean and store
    mean = acc_sum / F
    sumsq = acc_sumsq  # store in the output buffer for later std computation
    tl.store(mean_out_ptr + pid, mean)
    tl.store(sumsq_out_ptr + pid, sumsq)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for a single probability p_in[0].
    Uses Abramowitz & Stegun 7.1.26 approximation, with clamping to [eps, 1-eps].
    Writes the result to p_out_ptr[0].
    """
    p = tl.load(p_in_ptr)  # scalar read
    # Clamp to avoid log(0)/log(1)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Constants for approximation
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

    q = 0.0
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        numerator = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denominator = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        # atan2 approximation: atan(y) = atan(1/y) + pi/2 if y > 0 else pi/2 - atan(1/y)
        # Here use sign and reciprocal to emulate atan behavior
        sign = tl.where(q >= 0, 1.0, -1.0)
        # Compute atan via a simple path: atan(x) = 2 * atan(1/(2x + 1))
        # For q >= 0: atan(q) ≈ 2 * atan(1/(2q + 1)) = 2 * (pi/4 - atan((2q+1)/2)) approx, but simpler:
        # Use direct q mapping to avoid dependency on atan. Instead, rely on standard formula using sign.
        # For practical purposes, numerator/denominator gives sufficient accuracy; use sign adjustment.
        # However, we implement the A&S formula result directly:
        # result = sign * (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        result = sign * (numerator / denominator)
    else:
        # Central region: p in [p_low, p_high]
        q = p - 0.5
        r = q * q
        numerator = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denominator = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        result = numerator * q / denominator
        # Upper region: p > p_high
        if p > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            numerator_u = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            denominator_u = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
            result = - (numerator_u / denominator_u)

    # Store result to p_out_ptr[0]
    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply activation y = max(0, x - (mean[row] + std[row] * multiplier)),
    where mean_ptr/std_ptr are per-row scalars, multiplier_ptr is a 1-element tensor.
    """
    pid = tl.program_id(0)  # row id
    # Load per-row scalars
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    multiplier = tl.load(multiplier_ptr)  # scalar from 1-element tensor
    # Iterate over features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        idx = pid * F + cols
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        cutoff = mean + std * multiplier
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-7,
                 block_size: int = 1024, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Computes mean and std per (batch, seq) row across intermediate dimension.
        - Computes inverse CDF multiplier for target_sparsity in Triton.
        - Applies y = max(0, x - (mean + std * multiplier)) in Triton.
        Returns output cast to bfloat16.
        """
        # Ensure float32 for stable stats; we will cast output to bfloat16 at the end.
        x = inputs.to(torch.float32)
        B, S, F = x.shape
        rows = B * S

        # Flatten to [rows, F]
        x32 = x.reshape(rows, F)

        # 1) Compute per-row mean and sum of squares
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32,
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std = sqrt(sumsq/F - mean^2) using torch elementwise (Triton-only restriction)
        # This matches PyTorch's torch.std(..., unbiased=False) for last-dim reduction.
        var = sumsq / F - mean * mean
        std = torch.sqrt(var)  # 1D per-row std

        # 3) Compute inverse CDF (ndtri) for scalar target_sparsity in Triton
        # Create 1-element device tensor for input probability (no torch.tensor on tensors)
        p_in = x32.new_tensor(target_sparsity).contiguous()  # float32 scalar on device
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)

        relu_threshold_kernel[(rows,)](
            x32,
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
