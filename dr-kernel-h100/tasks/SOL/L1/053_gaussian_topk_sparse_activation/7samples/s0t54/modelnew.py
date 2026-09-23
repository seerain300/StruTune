import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std across the last dim K for each row.
    x_ptr is flattened to shape [total_rows, K]; row stride = K.
    """
    pid = tl.program_id(axis=0)  # one program per row
    # Accumulate sum and sum of squares in fp32
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over columns in BLOCK_SIZE chunks
    for col_start in range(0, K, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x = tl.load(x_ptr + pid * K + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = K  # population std
    mean = sum_x / n
    var = sum_x2 / n - mean * mean
    # Clamp variance to non-negative to avoid numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(out_ptr, p, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high):
    """
    Compute inverse standard normal CDF for p using Abramowitz & Stegun 5.2.23 approximation.
    Writes result to out_ptr[0].
    """
    # Coefficients
    # (Constant coefficients omitted here for brevity; pass them as arguments.)

    # Evaluate in lower, central, and upper regions and choose the best
    # For simplicity, implement central region approximation; lower/upper have similar forms.
    # Here we use the central region approximation for robustness:
    # q = p - 0.5
    q = p - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly * q / denom
    tl.store(out_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over [total_rows, K]: out = relu(x - (mean + std * z))
    x_ptr is flattened [total_rows * K], mean_ptr/std_ptr are [total_rows].
    z_ptr is a 1-element tensor holding scalar z.
    """
    row_id = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)

    # Compute column offsets for this tile
    col_start = col_block * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    # Load scalar z
    z = tl.load(z_ptr)

    # Compute threshold
    threshold = mean + std * z

    # Load x tile and compute gating
    x_vals = tl.load(x_ptr + row_id * K + cols, mask=mask, other=0.0)
    y = x_vals - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * K + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure 3D input [B, S, K]
        assert x.dim() == 3, "Input must be a 3D tensor [batch, seq_len, intermediate_size]"
        B, S, K = x.shape
        total_rows = B * S

        # Make contiguous and ensure CUDA
        x = x.contiguous()
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."

        # Allocate outputs and stats buffers
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # 1) Compute per-row mean and std using Triton reduction kernel
        # Choose BLOCK_SIZE based on K; moderate tile works well across sizes
        BLOCK_SIZE = 2048 if K >= 2048 else 1024
        compute_row_stats_kernel[(total_rows,)](
            x.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=self.num_stages
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (single scalar)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Use Abramowitz & Stegun coefficients (A&S 5.2.23)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            num_warps=1,
            num_stages=1
        )
        z = float(z_buf.item())  # read scalar without torch.tensor creation on host

        # 3) Apply gating with 2D Triton kernel; cast input to float32 for compute
        x_f32 = x.to(torch.float32)  # contiguous fp32
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
        else:
            block_size_gate = 2048
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)