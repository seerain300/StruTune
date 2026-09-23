import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std across the last dim K for each row.
    x_ptr is a flattened view of shape [total_rows, K], row stride = K.
    mean_ptr, std_ptr are shape [total_rows].
    """
    row_id = tl.program_id(0)
    # Bounds check: grid is exactly total_rows, so no mask needed.
    # Base pointer for this row
    base = row_id * K

    # Accumulate in fp32
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over columns in tiles of BLOCK_SIZE
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = tl.float32(K)
    mean = sum_x / n
    # Population std, unbiased=False
    var = sum_x2 / n - mean * mean
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity,  # output z_buf_ptr is scalar
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Compute z = _ndtri(target_sparsity) using Abramowitz & Stegun 5.2.23 approximation.
    Stores result to z_buf_ptr as float32.
    """
    # Only one program instance
    # Lower region
    # We'll do scalar computation; Triton expects vectors, but we can compute directly.
    # Implement piecewise approximation and store to z_buf_ptr
    # Note: Triton doesn't support 'if' with tensors, so we compute piecewise via masks and tl.where.
    # However, since we launch as a single program, we can do scalar math in registers.
    # We'll mimic the piecewise behavior with masks on scalar 'p' (target_sparsity).
    p = target_sparsity  # scalar
    q = 0.0

    # Scalar masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Compute q for each region
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # rational polynomial
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
    elif mask_mid:
        q = p - 0.5
        r = q * q
        poly = a1 * r + a2
        poly = poly * r + a3
        poly = poly * r + a4
        poly = poly * r + a5
        poly = poly * r + a6
        denom = b1 * r + b2
        denom = denom * r + b3
        denom = denom * r + b4
        denom = denom * r + b5
        denom = denom * r + 1.0
        z = poly * q / denom
    else:
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


# Autotuned gating kernel: 2D tiling over rows and columns
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
    ],
    key=['K'],
)
@triton.jit
def apply_gating_2d_kernel(inp_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Gating: out = relu(x - (mean + std * z)), where z = _ndtri(target_sparsity).
    inp_ptr, out_ptr: flattened [total_rows * K].
    mean_ptr, std_ptr: flattened [total_rows].
    z_ptr: scalar float32.
    """
    row_id = tl.program_id(0)
    col_tile = tl.program_id(1)
    col_start = col_tile * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)

    base = row_id * K
    x = tl.load(inp_ptr + base + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    threshold = mean + std * z
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Tuning knobs for kernels
        self.num_stages = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation of run(inputs, target_sparsity):
        - Compute per-row mean and std (population std, unbiased=False) along last dim.
        - Compute z = _ndtri(target_sparsity) via Triton approximation.
        - Apply gating: out = relu(x - (mean + std * z)).
        Returns tensor with same shape as x, dtype bfloat16.
        """
        # 1) Ensure contiguous input
        x_contig = x.contiguous()

        # Flattened rows
        B, S, K = x_contig.shape
        total_rows = B * S

        # 2) Allocate buffers for mean and std (fp32 for stability)
        mean = torch.empty((total_rows,), device=x.device, dtype=torch.float32)
        std = torch.empty((total_rows,), device=x.device, dtype=torch.float32)

        # Launch reduction kernel over flattened rows (grid: total_rows)
        grid_rows = (total_rows,)
        compute_row_stats_kernel[grid_rows](
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=1024,  # kernel loops over K; BLOCK_SIZE controls tile size
            num_warps=4, num_stages=self.num_stages
        )

        # 3) Compute z = _ndtri(target_sparsity) via Triton approximation (scalar)
        z_buf = torch.empty((1,), device=x.device, dtype=torch.float32)

        # Constants for A&S approximation (float32)
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
        # Launch Triton kernel; grid is (1,)
        compute_ndtri_kernel[(1,)](
            z_buf, self.target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            num_warps=1, num_stages=1
        )
        # Read scalar z without torch.tensor in host
        z = float(z_buf.item())

        # 4) Apply gating with Triton (autotuned 2D kernel)
        x_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        grid_gate = (total_rows, triton.cdiv(K, 1024))  # initial grid; autotune will pick best config
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean, std, torch.tensor(z, dtype=torch.float32, device=x.device),
            out_f32.view(-1), total_rows, K
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
