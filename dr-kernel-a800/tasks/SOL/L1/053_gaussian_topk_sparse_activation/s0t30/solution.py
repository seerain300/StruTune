import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = 0.0
    # Iterate features in chunks of BLOCK_F
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sum_sq = 0.0
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    var = sum_sq / F
    std = tl.sqrt(var)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(p, z_ptr,
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high):
    # Inverse normal CDF (A&S 5.2.23) for scalar p
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_c = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_d = ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = poly_c / poly_d

    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_a = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_b = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = poly_a * q_mid / poly_b

    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    poly_down = ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    z_up = poly_up / poly_down

    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_up, z)
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar z for this run
    cutoff = mean + std * z
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
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

        # Heuristics for block size and warps
        if F < 4096:
            BLOCK_F = 8192  # single loop iteration when F <= 4096
            num_warps = 8
            num_stages = 2
        elif F < 8192:
            BLOCK_F = 8192
            num_warps = 8
            num_stages = 2
        else:
            BLOCK_F = 16384  # 1 or 2 iterations for F up to 16384, good for 8192, 12288, 16384
            num_warps = 8
            num_stages = 2

        # Launch mean and std kernels: one program per row
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Device scalar for z (inverse-normal CDF)
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Triton-only ndtri approximation kernel (no torch.tensor creation on host)
        ndtri_approx_kernel[(1,)](
            float(target_sparsity), z_scalar,
            -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
            -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01,
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
            7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00,
            0.02425, 0.97575  # p_low and 1 - p_low
        )

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps, num_stages=num_stages)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
