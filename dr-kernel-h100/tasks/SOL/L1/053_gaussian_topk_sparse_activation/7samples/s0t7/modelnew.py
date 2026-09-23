import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_2d_kernel(x_ptr, mean_ptr, std_ptr,
                                B, S, K,
                                stride_row, stride_col,
                                BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std along last dimension (K) for each row (b, s).
    x is a flat contiguous tensor of size B*S*K. We interpret it with strides:
      - stride_row = K (since we flatten [B, S, K] and treat each row [s] as K elements)
      - stride_col = 1 (contiguous along last dim).
    mean_ptr and std_ptr are of length B*S, indexed by program_id(0).
    """
    row = tl.program_id(0)  # 0..B*S-1

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in chunks of BLOCK_SIZE
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # Address for this row: x_ptr + row * stride_row + cols * stride_col
        x = tl.load(x_ptr + row * stride_row + cols * stride_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = K
    mean = sum_val / n
    # Population std (unbiased=False)
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Store results to mean_ptr[row] and std_ptr[row]
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sp,
                          a1, a2, a3, a4, a5, a6,
                          b1, b2, b3, b4, b5,
                          c1, c2, c3, c4, c5, c6,
                          d1, d2, d3, d4,
                          p_low, p_high):
    """
    Compute inverse standard normal CDF (quantile function) at 'target_sp'
    using Abramowitz & Stegun 5.2.23 approximation and write to z_buf_ptr[0].
    """
    # piecewise constants
    inv_sqrt2 = 0.70710678118654752440  # 1/sqrt(2)
    p = target_sp  # should be in (0, 1)

    # lower region: p < p_low
    # q = sqrt(-2*log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # central region: p_low <= p <= p_high
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    y_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # upper region: p > p_high
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # select piecewise result
    # Triton supports where-like selection via comparisons and tl.where
    z = tl.where(p < p_low, y_low,
                 tl.where(p > p_high, y_up, y_mid))

    # write to z_buf_ptr[0]
    # We assume grid is (1,) so pointer arithmetic is simple
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z)), per row across K.
    x, out: flattened [total_rows * K] contiguous.
    mean, std: length total_rows (each corresponds to one row).
    z: scalar in z_ptr[0].
    """
    row = tl.program_id(0)  # 0..total_rows-1
    thresh = tl.load(mean_ptr + row) + tl.load(std_ptr + row) * tl.load(z_ptr)

    # Process columns in tiles
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0).to(tl.float32)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row * K + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size=2048, num_warps=4, num_stages=2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return for no sparsity
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and flatten
        x_contig = x.contiguous()
        B, S, K = x_contig.shape

        # Allocate mean and std as [B*S], will be viewed as [B, S, 1] later
        total_rows = B * S
        mean = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)

        # Launch reduction kernel: one program per row
        grid_stats = (total_rows,)
        compute_row_stats_2d_kernel[grid_stats](
            x_contig.view(-1),
            mean, std,
            B, S, K,
            stride_row=K, stride_col=1,  # since x is [B*S*K] contiguous and we treat each row as K elements
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Compute z = inverse normal CDF(target_sparsity) in Triton (no torch.tensor)
        # Prepare z buffer
        z_buf = torch.empty(1, dtype=torch.float32, device=x_contig.device)
        # Coefficients for Abramowitz & Stegun 5.2.23 approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00
        p_low = 0.02425; p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # host reads scalar; no torch.tensor creation

        # Prepare input and output as contiguous float32
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Launch gating kernel: grid over rows and column tiles
        grid_gate = (total_rows, triton.cdiv(K, self.block_size))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)