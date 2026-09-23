import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F, and sumsq[row].
    """
    row = tl.program_id(0)
    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over feature dimension in tiles
    for offs in range(0, F, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        base = row * F
        x = tl.load(X_ptr + base + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Write results
    mean_out_ptr[row] = sum_val / F
    sumsq_out_ptr[row] = sum_sq


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for the scalar p_in_ptr[0] using A&S 7.1.26 approximation.
    Writes result to p_out_ptr[0].
    """
    p = tl.load(p_in_ptr)
    # Clamp to [eps, 1-eps]
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
    low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    central = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
         ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high
    # Select piecewise result
    res = tl.where(mask_low, low, 0.0) + tl.where(mask_mid, central, 0.0) + tl.where(mask_high, up, 0.0)
    tl.store(p_out_ptr, res)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multipl_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, x - (mean[row] + std[row] * multipl_ptr[0])) elementwise for each row.
    X_ptr is [ROWS, F] flattened. Y_ptr is [ROWS*F].
    """
    row = tl.program_id(0)
    # Load scalar multiplier
    multiplier = tl.load(multipl_ptr)
    base = row * F

    # Loop over features in tiles
    for offs in range(0, F, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + base + cols, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + row)
        std = tl.load(std_ptr + row)
        cutoff = mean + std * multiplier
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self,
                 eps: float = 1e-7,
                 block_size: int = 2048,
                 num_warps: int = 8,
                 num_stages: int = 2):
        super().__init__()
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of run:
        - Computes per-row mean and std (unbiased=False) across last dim
        - Computes ndtri(target_sparsity) in Triton
        - Applies activation: y = max(0, x - (mean + std * ndtri))
        Returns y in bfloat16 (matching original output dtype).
        """
        # Ensure float32 for numerics
        x32 = x.to(torch.float32)
        B, S, F = x32.shape
        rows = B * S

        # Allocate outputs for stats
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        # 1) Compute per-row sum and sumsq via Triton
        grid = (rows,)
        row_stats_kernel[grid](
            x32, mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute per-row std on host (PyTorch) for unbiased=False semantics
        # var = E[x^2] - (E[x])^2
        var = sumsq / F - mean * mean
        std = torch.sqrt(var)

        # 3) Compute ndtri(target_sparsity) in Triton as a scalar
        p_in = x32.new_tensor(target_sparsity)  # 1-element device scalar (float32), no torch.tensor on tensors
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation via Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        relu_threshold_kernel[grid](
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