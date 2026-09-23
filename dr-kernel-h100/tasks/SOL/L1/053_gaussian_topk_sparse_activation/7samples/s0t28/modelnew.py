import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each row in [0, total_rows).
    x_ptr points to flattened input, stride per row is K.
    mean_ptr, std_ptr: shape [total_rows], float32.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    # Accumulators
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over K in chunks
    for start in range(0, K, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Each row's base is row_id * K
        vals = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(vals)
        sum_sq += tl.sum(vals * vals)

    n = tl.float32(K)
    mean = sum_val / n
    # population std, unbiased=False
    std = tl.sqrt(sum_sq / n - mean * mean)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity,  # z_buf_ptr is a 1-element buffer
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (z) for target_sparsity in (0,1).
    Uses Abramowitz & Stegun 5.2.23 approximation.
    Writes result to z_buf_ptr[0] as float32.
    """
    # Since grid is (1,), we can implement single-program computation.
    # Constants
    # We won't use p_low/p_high in this kernel (as it's scalar), but we keep them for API compatibility.

    # Implement the piecewise approximation:
    # Lower region
    # q = sqrt(-2 * log(p)) for p < p_low
    # Central region: p in [p_low, p_high]
    # Upper region: q = sqrt(-2 * log(1 - p)) for p > p_high
    # Then rational polynomial for central region.

    # Note: Triton does not support log of a tensor with mask; do scalar branchless via tl.where and per-element logicals.
    # However, here we only have one scalar 'target_sparsity'. We'll handle branches explicitly.

    # Assume target_sparsity is passed as float (0,1). If target_sparsity <= p_low: use lower approximation;
    # if target_sparsity >= p_high: use upper approximation; else central.

    # Compute q for lower and upper paths
    # For scalar, this is fine.
    if target_sparsity <= p_low:
        q = tl.sqrt(-2.0 * tl.log(target_sparsity))
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        den = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = poly / den
    elif target_sparsity >= p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - target_sparsity))
        poly = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        den = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = poly / den
    else:
        q = target_sparsity - 0.5
        r = q * q
        poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        den = (((((b1*r + b2)*r + b3)*r + b4)*r + b5))
        z = poly / den

    # Write result
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_perrow_kernel(x_ptr, mean_ptr, std_ptr, z_buf_ptr, out_ptr,
                               total_rows, K,
                               BLOCK_SIZE: tl.constexpr):
    """
    Per-row elementwise gating:
    For row i in [0, total_rows), compute threshold = mean[i] + std[i] * z_buf_ptr[0],
    then out[i*stride_out + j] = max(0, x[i*stride_in + j] - threshold).
    All pointers point to flattened contiguous buffers. We pass total_rows and K to index.
    """
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return

    # Load per-row mean and std and scalar z
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_buf_ptr)  # scalar

    threshold = mean + std * z

    # Iterate over K in chunks
    for start in range(0, K, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K

        x_row_ptr = x_ptr + row_id * K
        out_row_ptr = out_ptr + row_id * K

        x_vals = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        y_vals = x_vals - threshold  # broadcast scalar threshold
        y_vals = tl.maximum(y_vals, 0.0)  # ReLU
        tl.store(out_row_ptr + offs, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        x = x.contiguous()
        B, S, K = x.shape
        total_rows = B * S

        # 1) Compute per-row mean and std in fp32
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Choose BLOCK_SIZE for reduction; 1024 or 2048 are good defaults
        BLOCK_SIZE_STATS = 2048
        compute_row_stats_kernel[(total_rows,)](
            x.view(-1), mean, std, total_rows, K, BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=4, num_stages=2
        )

        # 2) Compute z = _ndtri(target_sparsity) in Triton (scalar)
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)

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

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            BLOCK_SIZE=1024, num_warps=1, num_stages=1
        )

        # 3) Apply gating per row with per-row program (loop over K)
        x_f32 = x.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)

        # Choose BLOCK_SIZE for per-row gating based on K for robustness
        if K >= 8192:
            BLOCK_SIZE_GATE = 4096
            num_warps_gate = 8
        else:
            BLOCK_SIZE_GATE = 2048
            num_warps_gate = 4

        apply_gating_perrow_kernel[(total_rows,)](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)