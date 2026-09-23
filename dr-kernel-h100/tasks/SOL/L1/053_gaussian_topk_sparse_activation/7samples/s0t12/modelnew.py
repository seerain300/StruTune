import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row (flattened index pid in 0..total_rows-1), compute:
      mean = sum(x[row, :]) / K
      std  = sqrt(sum(x[row, :]**2)/K - mean^2)
    Store mean[pid] and std[pid].
    x_ptr is assumed to point to a contiguous [B, S, K] flattened buffer.
    """
    pid = tl.program_id(0)  # 0..total_rows-1
    row = pid  # one program per row
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over K in chunks
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        # Pointer for this row
        x_row_ptr = x_ptr + row * K + cols
        vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)
    mean = sum_val / K
    var = sum_sq / K - mean * mean
    # Ensure numerical stability (var could be slightly negative due to rounding)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_ndtri_kernel(z_buf, target_sparsity,
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Compute z = ndtri(target_sparsity) using Abramowitz & Stegun 5.2.23 approximation.
    Store result in z_buf[0] as fp32.
    """
    p = target_sparsity  # scalar 0 < p < 1
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # Central region
    q2 = p - 0.5
    r = q2 * q2
    z_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q2 / \
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5) * r + 1.0)
    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1*q3 + c2)*q3 + c3)*q3 + c4)*q3 + c5)*q3 + c6) / \
             ((((d1*q3 + d2)*q3 + d3)*q3 + d4)*q3 + 1.0)

    # Choose the appropriate branch
    # Using where to pick value based on p
    # Note: Triton does not have a direct piecewise function, so select via conditions:
    cond_low = p < p_low
    cond_high = p > p_high
    # Default z_mid
    z = z_mid
    # If low, set z_low
    z = tl.where(cond_low, z_low, z)
    # If high, set z_high
    z = tl.where(cond_high, z_high, z)
    tl.store(z_buf, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = max(0, x - (mean + std * z))
    x_ptr: flattened [total_rows * K]
    mean_ptr, std_ptr: flattened [total_rows]
    z_ptr: single-element buffer containing z (fp32)
    out_ptr: flattened [total_rows * K]
    """
    row = tl.program_id(0)  # 0..total_rows-1
    col_block = tl.program_id(1)  # tile index along columns
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # Load per-row mean and std
    mean_val = tl.load(mean_ptr + row)
    std_val = tl.load(std_ptr + row)
    z_val = tl.load(z_ptr)  # scalar

    # Compute threshold
    threshold = mean_val + std_val * z_val

    # Load x for this row and cols
    x_row_ptr = x_ptr + row * K + cols
    x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)

    # Apply gating: relu(x - threshold)
    y = x_vals - threshold
    y = tl.maximum(y, 0.0)

    # Store
    out_row_ptr = out_ptr + row * K + cols
    tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for approximation (Abramowitz & Stegun 5.2.23)
        self.a1 = -3.969683028665376e+01
        self.a2 = 2.209460984245205e+02
        self.a3 = -2.759285104469687e+02
        self.a4 = 1.383577518672690e+02
        self.a5 = -3.066479806614716e+01
        self.a6 = 2.506628277459239e+00

        self.b1 = -5.447609879822406e+01
        self.b2 = 1.615858368580409e+02
        self.b3 = -1.556989798598866e+02
        self.b4 = 6.680131188771972e+01
        self.b5 = -1.328068155288572e+01

        self.c1 = -7.784894002430293e-03
        self.c2 = -3.223964580411365e-01
        self.c3 = -2.400758277161838e+00
        self.c4 = -2.549732539343734e+00
        self.c5 = 4.374664141464968e+00
        self.c6 = 2.938163982698783e+00

        self.d1 = 7.784695709041462e-03
        self.d2 = 3.224671290700398e-01
        self.d3 = 2.445134137142996e+00
        self.d4 = 3.754408661907416e+00

        self.p_low = 0.02425

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of:
          out = relu(x - (mean + std * ndtri(target_sparsity)))
        where mean/std are per [batch, seq] along the last dimension.
        Returns out in bfloat16.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and flatten for kernels
        x_contig = x.contiguous()
        B, S, K = x_contig.shape
        total_rows = B * S

        # 1) Compute per-row mean and std in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x_contig.device)

        # Choose BLOCK_SIZE for reduction; 1024 is a good default
        block_size_red = 1024
        compute_row_stats_kernel[(total_rows,)](
            x_contig.view(-1), mean, std, total_rows, K, BLOCK_SIZE=block_size_red,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = ndtri(target_sparsity) in Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=x_contig.device)
        # Pass target_sparsity as Python float (no torch tensor creation)
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            self.a1, self.a2, self.a3, self.a4, self.a5, self.a6,
            self.b1, self.b2, self.b3, self.b4, self.b5,
            self.c1, self.c2, self.c3, self.c4, self.c5, self.c6,
            self.d1, self.d2, self.d3, self.d4,
            self.p_low, 1.0 - self.p_low,
            num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # read scalar without torch tensor creation

        # 3) Apply gating with 2D Triton kernel
        x_f32 = x_contig.to(torch.float32)
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
            x_f32.view(-1), mean, std, torch.tensor([z], dtype=torch.float32, device=x_contig.device), out_f32.view(-1),
            total_rows, K, BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate, num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)