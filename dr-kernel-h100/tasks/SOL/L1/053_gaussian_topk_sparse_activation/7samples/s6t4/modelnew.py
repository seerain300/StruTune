import torch
import triton
import triton.language as tl


# Kernel: compute per-row (mean, std) across the last dimension (F)
@triton.jit
def _rowwise_stats_kernel(x_ptr, B, S, F,
                           means_ptr, stds_ptr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # one program per (b, s) row
    row_offset = pid * F
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, F, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < F
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    F_fp = tl.full((), F, tl.float32)
    mean = sum_val / F_fp
    # population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / F_fp - mean * mean
    # Avoid negative due to roundoff: clamp to >= 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Store results
    tl.store(means_ptr + pid, mean)
    tl.store(stds_ptr + pid, std)


# Kernel: compute inverse normal CDF for a scalar probability p using A&S 5.2.23
# Accepts a 1-element input tensor for p, writes a 1-element output tensor for icdf.
@triton.jit
def _icdf_ndtri_kernel(p_in_ptr, B, S, F, icdf_out_ptr):
    # Single program: compute icdf for one probability
    p = tl.load(p_in_ptr)  # scalar float
    # clamp to avoid log(0) or undefined regions
    p = tl.maximum(p, 1e-7)
    p = tl.minimum(p, 1.0 - 1e-7)

    # Constants for A&S 5.2.23
    p_low = 0.02425
    # p_high = 1.0 - p_low (not needed explicitly)
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

    # Region masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Lower region computation
    z_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * z_low + c2) * z_low + c3) * z_low + c4) * z_low + c5) * z_low + c6)
    denom_low = (((((d1 * z_low + d2) * z_low + d3) * z_low + d4) * z_low + 1.0))
    y_low = -poly_low / denom_low

    # Central region computation
    z_mid = p - 0.5
    r_mid = z_mid * z_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * z_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    y_mid = poly_mid / denom_mid

    # Upper region computation
    z_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * z_high + c2) * z_high + c3) * z_high + c4) * z_high + c5) * z_high + c6)
    denom_high = (((((d1 * z_high + d2) * z_high + d3) * z_high + d4) * z_high + 1.0))
    y_high = -poly_high / denom_high

    # Select region result
    y = tl.where(mask_low, y_low, 0.0)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_high, y_high, y)

    # Store result (1-element tensor)
    tl.store(icdf_out_ptr, y)


# Kernel: sparsify with ReLU threshold per row: out = max(0, x - (mean + std * icdf))
@triton.jit
def _sparsify_relu_kernel(x_ptr, B, S, F,
                           means_ptr, stds_ptr, icdf_ptr,
                           out_ptr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # one program per (b, s) row
    row_offset = pid * F

    # Load per-row stats and icdf
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    icdf = tl.load(icdf_ptr)  # scalar per forward

    threshold = mean + std * icdf

    # Apply ReLU on the row
    for start in range(0, F, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < F
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized forward:
        - Compute per-(batch,seq) mean and std across last dim (intermediate_size).
        - Compute inverse normal CDF for target_sparsity using A&S approximation in a Triton kernel.
        - Compute adaptive cutoff threshold and sparsify with ReLU.
        Returns: bfloat16 tensor of shape [B, S, F].
        """
        # Ensure contiguous for coalesced access
        x_fp32 = x.contiguous().to(torch.float32)
        B, S, F = x_fp32.shape

        # Output buffer in fp32 (for computation)
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=x_fp32.device)

        # Buffers for per-row stats
        means = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=x_fp32.device)

        # Launch rowwise stats kernel: one program per (b, s) row
        grid = (B * S,)
        _rowwise_stats_kernel[grid](
            x_fp32, B, S, F, means, stds,
            BLOCK=1024, num_warps=4
        )

        # Create device tensor for probability and run icdf kernel
        p_tensor = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        p_tensor.fill_(float(target_sparsity))

        icdf_tensor = torch.empty((), dtype=torch.float32, device=x_fp32.device)
        _icdf_ndtri_kernel[(1,)](
            p_tensor, B, S, F, icdf_tensor,
            BLOCK=1, num_warps=1
        )

        # Launch sparsification kernel: one program per (b, s) row
        _sparsify_relu_kernel[grid](
            x_fp32, B, S, F, means, stds, icdf_tensor, out_fp32,
            BLOCK=1024, num_warps=4
        )

        # Match original output dtype
        return out_fp32.to(torch.bfloat16)