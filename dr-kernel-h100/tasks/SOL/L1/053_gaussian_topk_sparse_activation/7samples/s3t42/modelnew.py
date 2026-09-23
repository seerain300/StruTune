import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Accumulate sum and sum of squares across F features.
    Writes mean[row] = sum/F, sumsq[row] where sumsq/F is used to compute variance.
    """
    row = tl.program_id(0)
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over the feature dimension in tiles
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        # Compute linear index for row
        idx = row * F + offs
        # Load with mask, other=0.0 to ignore out-of-bound elements
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Accumulate
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    # Write mean (sum/F)
    mean = total_sum / F
    tl.store(mean_out_ptr + row, mean)
    # Store sumsq for variance computation later
    tl.store(sumsq_out_ptr + row, total_sumsq)


@triton.jit
def compute_std_kernel(sumsq_ptr, mean_ptr, std_out_ptr,
                       F: tl.int32,
                       ROWS: tl.int32,
                       BLOCK_SIZE: tl.constexpr):
    """
    Elementwise compute std[row] = sqrt(sumsq[row]/F - mean[row]^2) for row in [0, ROWS).
    """
    row = tl.program_id(0)
    sumsq = tl.load(sumsq_ptr + row)
    mean = tl.load(mean_ptr + row)
    var = sumsq / F - mean * mean
    std = tl.sqrt(var)
    tl.store(std_out_ptr + row, std)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (quantile) of the single scalar in p_in_ptr
    using Abramowitz & Stegun 7.1.26 approximation and write to p_out_ptr.
    Clamp p to [eps, 1-eps] to avoid log(0)/log(1).
    """
    # Read input scalar
    p = tl.load(p_in_ptr)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)
    # Coefficients for A&S 7.1.26 approximation
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

    # Compute z for each region
    z_low = tl.sqrt(-2.0 * tl.log(p))
    z_mid = p - 0.5
    z_high = tl.sqrt(-2.0 * tl.log(1.0 - p))

    poly_low = (((((c1 * z_low + c2) * z_low + c3) * z_low + c4) * z_low + c5) * z_low + c6)
    poly_mid = (((((a1 * z_mid * z_mid + a2) * z_mid + a3) * z_mid + a4) * z_mid + a5) * z_mid + a6)
    poly_mid = poly_mid * z_mid
    poly_low_div = (((((d1 * z_low + d2) * z_low + d3) * z_low + d4) * z_low + 1.0))
    poly_high = -(((((c1 * z_high + c2) * z_high + c3) * z_high + c4) * z_high + c5) * z_high + c6)
    poly_high_div = (((((d1 * z_high + d2) * z_high + d3) * z_high + d4) * z_high + 1.0))

    # Select region
    # For scalar, tl.where supports scalar masks
    z = tl.where(mask_low, poly_low / poly_low_div, tl.where(mask_mid, poly_mid / z_mid, poly_high / poly_high_div))
    tl.store(p_out_ptr, z)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                          F: tl.int32, ROWS: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise activation per row:
    For each element x[row, f], compute y = max(0, x - (mean[row] + std[row] * multiplier)).
    """
    row = tl.program_id(0)
    # Load row params
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    m = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * m

    # Process features in tiles
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        idx = row * F + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.5, eps: float = 1e-7,
                 block_size: int = 1024, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.eps = float(eps)
        # Tunable parameters
        self.block_size = int(block_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [batch_size, seq_len, intermediate_size]
        Returns: same shape, bfloat16, with adaptive threshold sparsity.
        """
        B, S, F = inputs.shape
        x = inputs.to(torch.float32)
        rows = B * S

        # 1) Compute per-row sum and sumsq via Triton
        x_contig = x.contiguous()
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)

        grid = (rows,)
        row_stats_kernel[grid](
            x_contig.view(rows, F),
            mean, sumsq,
            F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=1,     # one program per row; reduction per program
            num_stages=2,
        )

        # 2) Compute std in Triton elementwise
        std = torch.empty(rows, dtype=torch.float32, device=x.device)
        compute_std_kernel[grid](
            sumsq, mean, std,
            F=F,
            ROWS=rows,
            BLOCK_SIZE=self.block_size,
            num_warps=1,     # simple elementwise per row
            num_stages=1,
        )

        # 3) Compute scalar multiplier (ndtri) via Triton scalar kernel
        p_in = x.new_tensor(self.target_sparsity)  # 1-element device scalar
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)
        relu_threshold_kernel[grid](
            x_contig.view(rows, F),
            mean, std, std_multiplier, y,
            F=F, ROWS=rows,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out