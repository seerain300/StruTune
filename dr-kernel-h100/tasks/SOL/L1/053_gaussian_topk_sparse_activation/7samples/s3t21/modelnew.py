import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes sum[row] and sumsq[row] to sum_out_ptr and sumsq_out_ptr (float32).
    """
    row_id = tl.program_id(axis=0)
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        row_base = row_id * F
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=0.0)
        # Accumulate in float32
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Write per-row results
    tl.store(sum_out_ptr + row_id, sum_val)
    tl.store(sumsq_out_ptr + row_id, sumsq_val)


@triton.jit
def var_std_kernel(sum_in_ptr, sumsq_in_ptr, mean_in_ptr, std_out_ptr,
                   ROWS: tl.int32, F: tl.int32):
    """
    Compute std per row: var = sumsq/F - mean^2; std = sqrt(max(var, 0)).
    Reads sum[row], sumsq[row], mean[row]; writes std[row].
    """
    row_id = tl.program_id(axis=0)
    sum_val = tl.load(sum_in_ptr + row_id)
    sumsq_val = tl.load(sumsq_in_ptr + row_id)
    mean_val = tl.load(mean_in_ptr + row_id)

    var = sumsq_val / F - mean_val * mean_val
    var = tl.maximum(var, 0.0)
    std_val = tl.sqrt(var)

    tl.store(std_out_ptr + row_id, std_val)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for the single element p_in_ptr[0].
    Uses Abramowitz & Stegun 7.1.26 approximation and clamps p to [eps, 1-eps].
    Writes result to p_out_ptr[0] as float32.
    """
    # Load probability and clamp
    p = tl.load(p_in_ptr)
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

    # Lower region
    mask_low = p < p_low
    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    # Upper region
    mask_high = p > p_high

    # Initialize result
    result = tl.zeros((), dtype=tl.float32)

    # Compute in each region
    # Lower region: q = sqrt(-2*log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    result_low = poly_low / den_low

    # Central region: q = p - 0.5, r = q^2
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    result_mid = poly_mid * q_mid / den_mid

    # Upper region: q = sqrt(-2*log(1-p))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    result_high = -poly_high / den_high

    # Select by mask: Triton will broadcast scalars; we can use tl.where
    result = tl.where(mask_low, result_low, result)
    result = tl.where(mask_mid, result_mid, result)
    result = tl.where(mask_high, result_high, result)

    tl.store(p_out_ptr, result)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          ROWS: tl.int32, F: tl.int32,
                          BLOCK_SIZE: tl.constexpr):
    """
    Apply activation y = max(0, x - (mean + std * multiplier)) for each element.
    X_ptr: input flattened [ROWS, F] (float32), read-only
    mean_ptr, std_ptr: per-row vectors (float32), read-only
    multiplier_ptr: 1-element tensor (float32), read-only
    Y_ptr: output flattened [ROWS, F] (float32), write
    """
    row_id = tl.program_id(axis=0)
    threshold = tl.load(mean_ptr + row_id) + tl.load(std_ptr + row_id) * tl.load(multiplier_ptr)

    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        row_base = row_id * F
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.5, eps: float = 1e-7,
                 block_size: int = 2048, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.eps = float(eps)
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, F], arbitrary strides; compute in float32
        B, S, F = x.shape
        rows = B * S

        # 1) Flatten and cast to float32 for stable statistics
        x32 = x.to(torch.float32).contiguous()  # ensure contiguous for simple pointer arithmetic

        # 2) Compute per-row sum and sumsq in Triton
        sum_out = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq_out = torch.empty(rows, dtype=torch.float32, device=x.device)

        # We need mean for computing std; compute mean in Triton as sum/F
        # But we only need sum and sumsq; mean can be derived later as sum/F
        row_stats_kernel[(rows,)](
            x32.view(rows, F),
            sum_out, sumsq_out,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 3) Prepare mean vector (sum/F)
        mean = sum_out / F

        # 4) Compute ndtri scalar multiplier in Triton
        p_in = x32.new_tensor(self.target_sparsity)  # 1-element device scalar (float32), no torch.tensor on tensors
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 5) Compute std per row in Triton: std = sqrt(sumsq/F - mean^2)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        var_std_kernel[(rows,)](
            sum_out, sumsq_out, mean, std,
            ROWS=rows, F=F,
            num_warps=4,
            num_stages=2,
        )

        # 6) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 7) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out