import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row (pid in 0..total_rows-1), compute:
      mean = sum(x[row, :]) / K
      std  = sqrt(sum(x[row, :]**2)/K - mean^2)  # population std (unbiased=False)
    Store mean[pid] and std[pid]. x_ptr points to a contiguous flattened [B, S, K] buffer.
    """
    pid = tl.program_id(0)  # each program handles one row
    # We don't have direct B,S here; x_ptr is contiguous flattened. Each row has K elements.
    # Reconstruct row pointer: row_start = pid * K
    # Note: Triton grid maps one program per row; total_rows is the number of rows (B*S).
    row_start = pid * K

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate across K in chunks of BLOCK_SIZE with masked loads
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        # x_vals are fp32
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity,  # scalar out, scalar in
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Compute inverse standard normal CDF z = _ndtri(p) using Abramowitz & Stegun 5.2.23 approximation.
    p is target_sparsity. Write result into z_buf_ptr[0].
    """
    p = target_sparsity  # scalar
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select piecewise (branchless selection via mask is fine here)
    # Piecewise logic based on p: p < p_low -> z_low; p > p_high -> z_up; else z_mid
    # Implement with masks
    mask_low = p < p_low
    mask_up = p > p_high
    z = tl.where(mask_low, z_low, z_mid)
    z = tl.where(mask_up, z_up, z)

    # Store single-element result
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_buf_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating per element: out = relu(x - (mean + std * z)).
    x_ptr, out_ptr: flattened pointers to input and output buffers (float32).
    mean_ptr, std_ptr: per-row stats (float32), length = total_rows.
    z_buf_ptr: single-element tensor with z (float32).
    Grid is (total_rows, ceil_div(K, BLOCK_SIZE)).
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid_row)
    std = tl.load(std_ptr + pid_row)
    # Load z scalar
    z = tl.load(z_buf_ptr)

    # Compute thresholds per element
    threshold = mean + std * z
    # Load x row
    x_row_start = pid_row * K
    x_vals = tl.load(x_ptr + x_row_start + cols, mask=mask, other=0.0)
    # Gating: relu(x - threshold)
    y = tl.maximum(x_vals - threshold, 0.0)
    # Store
    tl.store(out_ptr + pid_row * K + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 2048, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward:
        - Compute per-row mean and std (reduction).
        - Compute z = _ndtri(target_sparsity) in Triton.
        - Apply gating in Triton and return bfloat16.
        """
        # Ensure contiguous input
        x = inputs
        assert x.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        x = x.contiguous()
        B, S, K = x.shape
        total_rows = B * S

        # 1) Compute per-row mean and std
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Choose reduction BLOCK_SIZE; larger than K will still work but masked
        block_size_reduce = min(self.block_size, K)
        compute_row_stats_kernel[(total_rows,)](
            x.view(-1), mean, std, total_rows, K, BLOCK_SIZE=block_size_reduce,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton
        z_buf = torch.empty((), dtype=torch.float32, device=x.device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            num_warps=1, num_stages=1
        )

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x.to(torch.float32)  # compute in fp32
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K
        block_size_gate = self.block_size if K >= self.block_size else K
        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K, BLOCK_SIZE=block_size_gate,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)