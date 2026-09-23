import torch
import triton
import triton.language as tl


@triton.jit
def _ndtri_kernel(p_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (ndtri) for p_ptr[0].
    Implements Abramowitz and Stegun 7.1.26 piecewise rational approximation.
    Writes result to out_ptr[0].
    """
    p = tl.load(p_ptr)  # scalar float32
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

    # Compute z based on piecewise region
    # Masks as vectors of length 1 for scalar p
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_denom = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / poly_denom

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select by mask (scalar case; Triton handles elementwise ops)
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    # Store to out_ptr[0]
    tl.store(out_ptr, z)


@triton.jit
def _row_stats_kernel(X_ptr, mean_ptr, var_ptr, ROWS: tl.int32, F: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    For each row r in [0, ROWS), compute mean and variance along the last dim F.
    X_ptr is laid out as [ROWS, F] logically; strides are not used here since we assume contiguous [B, S, F].
    We pass ROWS and F; actual strides are handled via indexing r * F + col.
    """
    r = tl.program_id(axis=0)
    # If grid is larger than rows, guard (not necessary if grid==ROWS)
    # Compute sum and sumsq
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over F in tiles
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        # Row base index (assuming x is contiguous [B, S, F] and we pass ROWS as B*S)
        # X_ptr indexing: element at (r, col) = base + r*F + col
        # We create row pointer by assuming contiguous layout [ROWS, F]
        # Here, X_ptr is actually a flattened [ROWS, F], so pointer is X_ptr + r*F + offs
        row_base = r * F
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
        # Accumulate sums
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and variance (unbiased=False)
    mean = sum_val / F
    var = sum_sq / F - mean * mean  # E[x^2] - (E[x])^2
    # Store
    tl.store(mean_ptr + r, mean)
    tl.store(var_ptr + r, var)


@triton.jit
def _relu_threshold_kernel(X_ptr, mean_ptr, var_ptr, multiplier_ptr, Y_ptr,
                            ROWS: tl.int32, F: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Apply y = max(0, X - (mean + sqrt(var) * multiplier)) elementwise per row.
    X is assumed contiguous [ROWS, F]. Y is output [ROWS, F] float32.
    """
    r = tl.program_id(axis=0)
    if r >= ROWS:
        return

    # Load mean and var for row r
    mean = tl.load(mean_ptr + r)
    var = tl.load(var_ptr + r)
    std = tl.sqrt(var)
    multiplier = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * multiplier

    # Compute y for the row
    for start in range(0, F, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < F
        row_base = r * F
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + row_base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 4):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [B, S, F] input tensor (float32 recommended). Any dtype is fine; we compute in float32.
        Returns bfloat16 tensor with ReLU thresholding per [b, s] row using adaptive cutoff.
        """
        # Ensure we work with float32 for numerical stability; keep original dtype for output cast
        x_f32 = x.to(torch.float32)

        # Flatten to [ROWS, F] where ROWS = B * S
        B, S, F = x_f32.shape
        rows = B * S

        # Allocate outputs for stats and final activation
        mean = torch.empty(rows, dtype=torch.float32, device=x_f32.device)
        var = torch.empty(rows, dtype=torch.float32, device=x_f32.device)
        y = torch.empty(rows * F, dtype=torch.float32, device=x_f32.device)  # flattened [ROWS, F]

        # 1) Compute row-wise mean and var via Triton
        _row_stats_kernel[(rows,)](
            x_f32.view(-1),  # flatten to [ROWS*F], but kernel assumes [ROWS, F] with strides, so we pass view(ROWS, F)
            mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std_multiplier using Triton kernel (scalar evaluation of ndtri)
        # Create a 1-element tensor for p and output; Triton will read/write scalar
        p_tensor = torch.empty(1, dtype=torch.float32, device=x_f32.device)  # dummy tensor; we pass scalar via pointer arithmetic in kernel
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch ndtri kernel; we pass p=0.5 into p_tensor[0] before launch
        with torch.no_grad():
            p_tensor.fill_(target_sparsity)
        _ndtri_kernel[(1,)](
            p_tensor, std_multiplier,
            num_warps=1,
            num_stages=1,
        )

        # 3) Apply ReLU threshold in Triton
        _relu_threshold_kernel[(rows,)](
            x_f32.view(rows, F),
            mean, var, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Reshape y to [B, S, F] and cast to bfloat16 to match original
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out