import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std (population std, unbiased=False) along last dim K.
    x_ptr points to flattened [total_rows*K] row-major data with stride K between rows.
    Writes mean and std as [total_rows] float32.
    """
    row_id = tl.program_id(0)
    # If grid > total_rows, mask out
    if row_id >= total_rows:
        return

    # Accumulators in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # Iterate over K in tiles
    for start in range(0, K, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Address for this row: base = row_id * K, then + offs
        x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = tl.float32(K)
    mean = sum_x / n
    var = sum_x2 / n - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to fp errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, p, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for p in (0,1).
    Uses Abramowitz and Stegun 5.2.23 approximation.
    Writes a single float to z_buf_ptr[0].
    """
    # Single program instance
    q = 0.0
    # Lower region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = c1 * q + c2
        poly = poly * q + c3
        poly = poly * q + c4
        poly = poly * q + c5
        poly = poly * q + c6
        denom = d1 * q + d2
        denom = denom * q + d3
        denom = denom * q + d4
        denom = denom * q + 1.0
        z = poly / denom
    else:
        # Central region
        q = p - 0.5
        r = q * q
        poly_num = a1 * r + a2
        poly_num = poly_num * r + a3
        poly_num = poly_num * r + a4
        poly_num = poly_num * r + a5
        poly_num = poly_num * r + a6
        poly_den = b1 * r + b2
        poly_den = poly_den * r + b3
        poly_den = poly_den * r + b4
        poly_den = poly_den * r + b5
        num = poly_num * q
        den = poly_den * r + 1.0
        z = num / den
        # Upper region
        if p > p_high:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = c1 * q + c2
            poly = poly * q + c3
            poly = poly * q + c4
            poly = poly * q + c5
            poly = poly * q + c6
            denom = d1 * q + d2
            denom = denom * q + d3
            denom = denom * q + d4
            denom = denom * q + 1.0
            z = -poly / denom
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_scalar_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: out[b*s, k] = relu(x[b*s, k] - (mean[b*s] + std[b*s] * z))
    x_ptr: flattened [total_rows*K], fp32
    mean_ptr, std_ptr: [total_rows], fp32
    z_scalar_ptr: [1], fp32 scalar
    out_ptr: flattened [total_rows*K], fp32
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= total_rows:
        return

    start = col_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load row stats
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_scalar_ptr)  # scalar float

    # Compute threshold
    threshold = mean + std * z

    # Load x row segment, compute, store
    x_row_start = x_ptr + row * K
    x_vals = tl.load(x_row_start + offs, mask=mask, other=0.0)
    # x_vals is fp32
    y = x_vals - threshold
    y = tl.maximum(y, 0.0)  # relu
    out_row_start = out_ptr + row * K
    tl.store(out_row_start + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for A&S 5.2.23 approximation
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
        Triton-only implementation of the original logic:
        - Compute per-row mean and std (population std, unbiased=False) across last dim K.
        - Compute z = inverse normal CDF of target_sparsity using A&S 5.2.23.
        - Apply gating: out = relu(x - (mean + std * z)), returned as bfloat16.
        """
        if target_sparsity == 0.0:
            return x  # no sparsity requested

        # Ensure CUDA and contiguous layout
        assert x.is_cuda, "ModelNew requires CUDA tensors."
        x_contig = x.contiguous()
        B, S, K = x_contig.shape
        total_rows = B * S

        # 1) Allocate stats buffers and compute row stats
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Choose BLOCK_SIZE for reduction. 1024 works well across sizes.
        BLOCK_SIZE = 1024
        grid = (total_rows,)
        compute_row_stats_kernel[grid](x_contig.view(-1), mean, std, total_rows, K, BLOCK_SIZE,
                                       num_warps=4, num_stages=2)

        # 2) Compute z = _ndtri(target_sparsity) via Triton kernel
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            self.a1, self.a2, self.a3, self.a4, self.a5, self.a6,
            self.b1, self.b2, self.b3, self.b4, self.b5,
            self.c1, self.c2, self.c3, self.c4, self.c5, self.c6,
            self.d1, self.d2, self.d3, self.d4,
            self.p_low, 1.0 - self.p_low,
            BLOCK_SIZE=1024, num_warps=1, num_stages=1
        )

        # 3) Apply gating in Triton over [total_rows, K] with 2D grid
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K for robustness
        if K >= 16384:
            block_size_gate = 4096
            num_warps_gate = 4
        elif K >= 8192:
            block_size_gate = 2048
            num_warps_gate = 4
        else:
            block_size_gate = 1024
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)