import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(x_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    total = tl.zeros((), dtype=tl.float32)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    mean = total / F
    tl.store(means_ptr + pid, mean)


@triton.jit
def std_lastdim_kernel(x_ptr, stds_ptr, means_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        diff = vals - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    # population std: divide by F
    std = tl.sqrt(sum_sq / F)
    tl.store(stds_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(sp, z_ptr):
    # Compute inverse normal CDF (quantile) for scalar sp using A&S 5.2.23.
    # sp: scalar float32
    # z_ptr: [1] device tensor to store scalar z (float32)
    # Constants
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

    # lower tail
    mask_low = sp < p_low
    q_low = tl.sqrt(-2.0 * tl.log(sp))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))

    # central region
    mask_mid = (sp >= p_low) & (sp <= p_high)
    q_mid = sp - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # upper tail
    mask_high = sp > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - sp))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
              (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))

    # select appropriate branch
    z = tl.where(sp < p_low, z_low, tl.where(sp > p_high, z_high, z_mid))
    tl.store(z_ptr, z)


@triton.jit
def apply_cutoff_relu_kernel(x_ptr, out_ptr, means_ptr, stds_ptr, z_ptr, B, S, F, BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_offset = pid * F
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    cutoff = mean + std * z
    for offs in range(0, F, BLOCK_F):
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        res = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + row_offset + idx, res, mask=mask)


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
            BLOCK_F = 2048
            num_warps = 4
        elif F < 8192:
            BLOCK_F = 4096
            num_warps = 8
        else:
            BLOCK_F = 8192
            num_warps = 8

        # Launch mean and std kernels
        grid = (total_rows,)
        mean_lastdim_kernel[grid](x, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)
        std_lastdim_kernel[grid](x, stds, means, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # 1-element device tensor for z, filled by ndtri kernel
        z_scalar = torch.empty(1, device=x.device, dtype=torch.float32)

        # Launch ndtri approximation kernel with scalar target_sparsity (no torch.tensor on host)
        ndtri_approx_kernel[(1,)](target_sparsity, z_scalar)

        # Allocate output (float32) for apply kernel
        out_fp32 = torch.empty_like(x, dtype=torch.float32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](x, out_fp32, means, stds, z_scalar, B, S, F, BLOCK_F=BLOCK_F, num_warps=num_warps)

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
