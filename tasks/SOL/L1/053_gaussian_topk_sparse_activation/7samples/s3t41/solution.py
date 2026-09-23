import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F and stores sumsq[row] for later use.
    """
    row = tl.program_id(0)
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)
    # Loop over feature dimension in tiles
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        ptr = X_ptr + row * F + offs
        x = tl.load(ptr, mask=mask, other=0.0)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)
    mean = total_sum / F
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, total_sumsq)


@triton.jit
def std_kernel(sumsq_ptr, mean_ptr, std_out_ptr,
               ROWS: tl.int32, F: tl.int32):
    """
    For each row, compute std[row] = sqrt(sumsq[row]/F - mean[row]^2)
    and write to std_out_ptr[row].
    """
    row = tl.program_id(0)
    sumsq = tl.load(sumsq_ptr + row)
    m = tl.load(mean_ptr + row)
    var = sumsq / F - m * m
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr,
                        eps: tl.float32):
    """
    Compute inverse standard normal CDF (A&S 7.1.26) for p_in_ptr[0].
    Write result to p_out_ptr[0].
    Clamp p to [eps, 1-eps] to avoid log(0)/log(1).
    """
    # Read input scalar
    p = tl.load(p_in_ptr)
    # Clamp
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
    mask_low = p < p_low
    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    # Upper region
    mask_high = p > p_high

    # For these masks, we need to compute piecewise result. We can use tl.where.
    # Note: For upper region, since p>1-p_low, use 1 - p and negate final result.
    # We compute q per region:
    # Lower: q = sqrt(-2*log(p))
    # Mid: q = p - 0.5
    # Upper: q = sqrt(-2*log(1 - p))

    # Compute q_mid for central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    poly_mid = num_mid / den_mid

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
               ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Upper region: q = sqrt(-2*log(1 - p)); result is -poly
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Assemble final result based on masks
    # Initialize result to 0
    result = tl.zeros((), dtype=tl.float32)
    # Mid
    result = tl.where(mask_mid, poly_mid, result)
    # Low
    result = tl.where(mask_low, poly_low, result)
    # High
    result = tl.where(mask_high, poly_high, result)

    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    For each row, load mean[row], std[row], and multiplier[0].
    Compute cutoff = mean + std * multiplier.
    For all elements in the row, y = max(0, X - cutoff).
    """
    row = tl.program_id(0)
    m = tl.load(mean_ptr + row)
    s = tl.load(std_ptr + row)
    mult = tl.load(multiplier_ptr)  # scalar
    cutoff = m + s * mult
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        x = tl.load(X_ptr + row * F + offs, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + row * F + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.05, eps: float = 1e-7,
                 block_size: int = 1024, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure float32 contiguous
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.float()
        B, S, F = x.shape
        rows = B * S

        # 1) Compute per-row sums and sum of squares via Triton
        sum_ptr = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_ptr = torch.empty(rows, dtype=torch.float32, device=x.device)

        row_stats_kernel[(rows,)](
            x.view(rows, F),
            sum_ptr, sumsq_ptr,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute mean = sum/F
        # 3) Compute std = sqrt(sumsq/F - mean^2) via Triton kernel
        mean_ptr = torch.empty(rows, dtype=torch.float32, device=x.device)
        std_ptr = torch.empty(rows, dtype=torch.float32, device=x.device)

        # mean = sum / F (vector operation on CPU for simplicity, since it's small)
        mean_ptr.copy_(sum_ptr / F)

        std_kernel[(rows,)](
            sumsq_ptr, mean_ptr, std_ptr,
            ROWS=rows, F=F,
            num_warps=2,  # small, elementwise
            num_stages=1,
        )

        # 4) Compute scalar multiplier via Triton ndtri approximation
        p_in = x.new_tensor(self.target_sparsity)  # 1-element device scalar
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 5) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)

        relu_threshold_kernel[(rows,)](
            x.view(rows, F),
            mean_ptr, std_ptr, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 6) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
