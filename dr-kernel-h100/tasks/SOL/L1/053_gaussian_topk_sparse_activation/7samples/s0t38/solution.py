import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row (flattened [total_rows]).
    x_ptr: flattened input pointer, stride per row is K.
    mean_ptr, std_ptr: shape [total_rows]
    """
    row_id = tl.program_id(axis=0)
    # Compute row base offset: row_id * K
    offs = row_id * K
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)
    count = 0.0

    # Loop over columns in tiles of BLOCK_SIZE
    for start in range(0, K, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        # Load a tile of the row; masked elements load 0
        x = tl.load(x_ptr + offs + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Accumulate sum and sum of squares
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        # Count number of valid elements
        count += tl.sum(mask.to(tl.float32), axis=0)

    mean = acc_sum / K
    # population std (unbiased=False): std = sqrt(E[x^2] - (E[x])^2)
    var = acc_sumsq / K - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high, BLOCK_SIZE: tl.constexpr):
    """
    Compute z = inverse standard normal CDF for a single scalar target_sparsity using A&S 5.2.23 approximation.
    Write result to z_buf_ptr[0] as float32.
    """
    # Single-program computation
    q = target_sparsity
    # Lower region
    mask_low = q < p_low
    # Central region
    mask_mid = (q >= p_low) & (q <= p_high)
    # Upper region
    mask_high = q > p_high

    # Initialize result
    result = tl.zeros((), dtype=tl.float32)

    # Lower region approximation
    if mask_low:
        u = tl.sqrt(-2.0 * tl.log(q))
        poly = (((((c1*u + c2)*u + c3)*u + c4)*u + c5)*u + c6)
        poly_over = ((((d1*u + d2)*u + d3)*u + d4)*u + 1.0)
        result = poly / poly_over

    # Central region approximation
    if mask_mid:
        q_center = q - 0.5
        r = q_center * q_center
        poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        poly_over = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        result = poly * q_center / poly_over

    # Upper region approximation
    if mask_high:
        u = tl.sqrt(-2.0 * tl.log(1.0 - q))
        poly = (((((c1*u + c2)*u + c3)*u + c4)*u + c5)*u + c6)
        poly_over = ((((d1*u + d2)*u + d3)*u + d4)*u + 1.0)
        result = -poly / poly_over

    tl.store(z_buf_ptr, result)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating elementwise: out = relu(x - (mean + std * z)), broadcast mean/std per row.
    x_ptr: flattened [total_rows*K] float32
    mean_ptr, std_ptr: [total_rows] float32
    z_ptr: [1] float32 scalar
    out_ptr: flattened [total_rows*K] float32
    """
    row_id = tl.program_id(axis=0)
    col_tile = tl.program_id(axis=1)
    start = col_tile * BLOCK_SIZE
    idx = start + tl.arange(0, BLOCK_SIZE)
    mask = idx < K

    # Load row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar

    # Compute threshold per element
    threshold = mean + std * z
    x = tl.load(x_ptr + row_id * K + idx, mask=mask, other=0.0)
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row_id * K + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float = 0.0) -> torch.Tensor:
        # Ensure target_sparsity is in [0,1]; behavior matches original: if 0, return x unchanged
        if target_sparsity == 0.0:
            return x

        # Flatten [B, S, K] to [total_rows, K] for per-row stats and gating
        B, S, K = x.shape
        total_rows = B * S

        # Make input contiguous along the last dim
        x_contig = x.contiguous()
        # Allocate per-row mean and std as float32
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # 1) Compute per-row mean and std in Triton
        BLOCK_SIZE = 4096  # large tile for reduction across K
        grid = (total_rows,)
        compute_row_stats_kernel[grid](
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        a1, a2, a3, a4, a5, a6 =  -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
        b1, b2, b3, b4, b5 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
        c1, c2, c3, c4, c5, c6 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
        d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024,
            num_warps=1, num_stages=1
        )

        z = float(z_buf.item())  # single scalar load

        # 3) Apply gating with 2D Triton kernel (large tiles for better throughput)
        x_f32 = x_contig.to(torch.float32)  # compute in fp32 for stability
        out_f32 = torch.empty_like(x_f32)  # ensure dtype and shape match x_f32

        # Dynamic tuning for gating kernel: use larger tiles for large K
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
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
