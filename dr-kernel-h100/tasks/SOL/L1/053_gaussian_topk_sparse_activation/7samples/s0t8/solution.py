import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_2d_kernel(x_ptr, mean_ptr, std_ptr,
                                total_rows, col_stride, K,
                                BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (per [batch, seq]) mean and std along the last dimension (K).
    x_ptr: flattened view of input [B*S*K]. We index each row as a contiguous slice of length K.
    mean_ptr/std_ptr: shape [total_rows] (we'll view as [B, S, 1] in host).
    total_rows = B * S, col_stride = K (since rows are contiguous across columns).
    """
    row = tl.program_id(0)  # 0..total_rows-1
    sum_val = 0.0
    sum_sq = 0.0

    for off in range(0, K, BLOCK_SIZE):
        cols = off
        idx = row * col_stride + cols + tl.arange(0, BLOCK_SIZE)
        mask = cols + tl.arange(0, BLOCK_SIZE) < K
        # Load a chunk of the row into fp32
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    K_fp = tl.full((), K, tl.float32)
    mean = sum_val / K_fp
    # population std (unbiased=False): std = sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / K_fp - mean * mean
    # Ensure non-negative due to numerical noise
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store mean and std to output buffers
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4,
                         p_low, p_high):
    """
    Compute inverse standard normal CDF (quantile) z for p=target_sparsity using A&S 5.2.23.
    Write result to z_buf_ptr[0] (fp32).
    """
    # Load target_sparsity as fp32 scalar
    p = target_sparsity

    # Initialize piecewise branches
    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_low = poly_low / ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = -poly_low

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid_num = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_mid_den = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid_num * q_mid / poly_mid_den

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    poly_high = poly_high / ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = poly_high  # positive since 1-p is small

    # Select piecewise
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    # Select z based on region
    z = tl.where(cond_low, z_low, tl.where(cond_mid, z_mid, z_high))

    # Store into z_buf_ptr[0]
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out = relu(x - (mean + std * z)),
    where mean and std are per-row (loaded from mean_ptr, std_ptr).
    total_rows = B * S, K is last-dim size.
    """
    pid_row = tl.program_id(0)  # 0..total_rows-1
    pid_col = tl.program_id(1)  # 0..ceil_div(K, BLOCK_SIZE)-1
    col_start = pid_col * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid_row)
    std = tl.load(std_ptr + pid_row)
    z = tl.load(z_ptr)  # scalar

    # Compute thresholds
    threshold = mean + std * z
    # Load chunk of x
    row_start = pid_row * K
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    # Store
    tl.store(out_ptr + row_start + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 0, num_warps: int = 0, num_stages: int = 2):
        super().__init__()
        # If not provided, we will dynamically choose in forward
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [B, S, K] input tensor (any dtype, typically bfloat16)
        target_sparsity: float in (0,1), e.g., 0.1
        Returns: [B, S, K] tensor in bfloat16 after gating.
        """
        # Ensure 3D input
        assert x.dim() == 3, "Input must be 3D [batch, seq, features]"
        B, S, K = x.shape
        total_rows = B * S

        # Make contiguous for predictable strides
        x_contig = x.contiguous()
        # Allocate per-row stats as fp32
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Dynamically choose BLOCK_SIZE and num_warps if not provided
        block_size = self.block_size if self.block_size > 0 else (4096 if K >= 8192 else (2048 if K >= 4096 else 1024))
        num_warps = self.num_warps if self.num_warps > 0 else (4 if block_size >= 4096 else 4)

        # Launch reduction kernel: one program per row
        grid_stats = (total_rows,)
        compute_row_stats_2d_kernel[grid_stats](
            x_contig, mean, std,
            total_rows, K, K,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=self.num_stages
        )

        # Allocate scalar z buffer in fp32
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Launch ndtri kernel: pass coefficients and p
        # Coefficients from Abramowitz & Stegun 5.2.23
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            num_warps=1, num_stages=1
        )

        # Prepare input and output as contiguous float32 for computation
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Launch gating kernel: grid over rows and column tiles
        grid_gate = (total_rows, triton.cdiv(K, block_size))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
