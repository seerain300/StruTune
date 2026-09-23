import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # 2D grid: axis 0 = batch, axis 1 = seq
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    row_offset = (b * S + s) * F
    total = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    # Store per-row mean in flat [B*S] order
    tl.store(means_ptr + b * S + s, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    row_offset = (b * S + s) * F
    mean = tl.load(means_ptr + b * S + s)
    total = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = x - mean
        total += tl.sum(diff * diff, axis=0)
    var = total / F  # population variance
    std = tl.sqrt(var)
    tl.store(stds_ptr + b * S + s, std)


@triton.jit
def ndtri_approx_kernel(p_scalar, z_ptr,
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4):
    # Piecewise Abramowitz & Stegun 5.2.23 approximation
    p = p_scalar
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_num = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_den = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_num / poly_den

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_a = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_b = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_a * q_mid / poly_b

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_numh = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    poly_denh = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_numh / poly_denh

    # Select region
    if p < p_low:
        z = z_low
    elif p <= p_high:
        z = z_mid
    else:
        z = z_high

    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    # 2D grid: axis 0 = batch, axis 1 = seq
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    row_offset = (b * S + s) * F
    mean = tl.load(means_ptr + b * S + s)
    std = tl.load(stds_ptr + b * S + s)
    z = tl.load(z_ptr)
    cutoff = mean + std * z
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_offset + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)
        B, S, F = x.shape
        total_rows = B * S

        # Allocate per-row vectors
        means = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        stds = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Heuristics for block size and warps to minimize loop iterations
        if F <= 4096:
            BLOCK_F = 4096
            num_warps = 8
            num_stages = 2
        elif F <= 8192:
            BLOCK_F = 8192
            num_warps = 8
            num_stages = 2
        else:
            BLOCK_F = 16384
            num_warps = 8
            num_stages = 2

        # 2D grid for mean and std kernels: (B, S)
        grid2d = (B, S)
        mean_lastdim_kernel[grid2d](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
        std_lastdim_kernel[grid2d](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Device scalar for z (inverse-normal CDF)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Triton-only ndtri approximation kernel (no torch.tensor creation on host)
        ndtri_approx_kernel[(1,)](
            float(target_sparsity), z_scalar,
            -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
            -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01,
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
            7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
        )

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # 2D grid for apply kernel
        apply_cutoff_relu_kernel[grid2d](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
