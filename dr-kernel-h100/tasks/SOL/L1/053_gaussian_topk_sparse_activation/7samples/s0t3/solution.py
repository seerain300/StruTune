import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d_kernel(
    x_ptr,               # *float32, shape [rows, K], contiguous
    mean_ptr,            # *float32, shape [rows], contiguous
    std_ptr,             # *float32, shape [rows], contiguous
    rows,                # int32, number of rows (B*S)
    K,                   # int32, number of columns (intermediate_size)
    BLOCK_SIZE: tl.constexpr
):
    # One program per row
    row = tl.program_id(0)
    # If row >= rows, exit (grid should match, but guard anyway)
    if row >= rows:
        return

    # Accumulators for sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over columns in chunks
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Compute pointer for this row and chunk
        x_row_ptr = x_ptr + row * K
        x_chunk = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        # Accumulate
        sum_val += tl.sum(x_chunk, axis=0)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    # Compute mean and std (population std)
    K_f = tl.full((), K, tl.float32)
    mean = sum_val / K_f
    var = sum_sq / K_f - mean * mean
    # var can be negative due to numerical error; clamp to 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def apply_gating_2d_kernel(
    x_ptr,               # *float32, shape [rows, K], contiguous
    mean_ptr,            # *float32, shape [rows], contiguous
    std_ptr,             # *float32, shape [rows], contiguous
    out_ptr,             # *float32, shape [rows, K], contiguous
    z,                   # float32 scalar, _ndtri(target_sparsity)
    rows,                # int32
    K,                   # int32
    BLOCK_SIZE: tl.constexpr
):
    # 2D launch: grid = (rows, ceil_div(K, BLOCK_SIZE))
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= rows:
        return

    col_start = col_block * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load x chunk
    x_row_ptr = x_ptr + row * K
    x_chunk = tl.load(x_row_ptr + offs, mask=mask, other=0.0)

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    threshold = mean + std * z
    # Gate: relu(input - threshold)
    gated = x_chunk - threshold
    # Ensure broadcasting: threshold is scalar
    gated = tl.maximum(gated, 0.0)

    # Store
    out_row_ptr = out_ptr + row * K
    tl.store(out_row_ptr + offs, gated, mask=mask)


def _ndtri_torch(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function) using A&S 5.2.23 approximation.
    This is a helper for host-side computation of z; we will use it to obtain the scalar z.
    """
    # Constants for A&S 5.2.23
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

    # Central region
    q = p - 0.5
    r = q * q
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    result = (poly * q) / denom

    # Lower and upper tails
    # If p < p_low
    if p < p_low:
        t = torch.sqrt(-2.0 * torch.log(p))
        result = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6) / \
                 ((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0)
    elif p > p_high:
        t = torch.sqrt(-2.0 * torch.log(1.0 - p))
        result = -(((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6) / \
                 ((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0)

    return float(result)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run.
        - Computes per-row mean and std using Triton reduction kernel
        - Computes z = _ndtri(target_sparsity) on host
        - Applies gating: out = relu(input - (mean + std * z)) using Triton elementwise kernel
        Returns: tensor of same shape as inputs, dtype bfloat16
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and work in float32 for numerical stability
        x = inputs.contiguous().to(torch.float32)

        B, S, K = x.shape
        rows = B * S

        # 2D views: [rows, K]
        x2d = x.view(rows, K)

        # Allocate mean and std buffers [rows]
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per row
        grid_reduce = (rows,)
        reduce_mean_std_2d_kernel[grid_reduce](
            x2d, mean, std,
            rows, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Compute z = _ndtri(target_sparsity) on host
        z = _ndtri_torch(float(target_sparsity))

        # Prepare output buffer [rows, K] in float32
        out2d = torch.empty_like(x2d)

        # Launch gating kernel over columns
        grid_gate = (rows, triton.cdiv(K, self.block_size))
        apply_gating_2d_kernel[grid_gate](
            x2d, mean, std, out2d,
            z,
            rows, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Reshape back to [B, S, K] and cast to bfloat16 to match original behavior
        out = out2d.view(B, S, K).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
