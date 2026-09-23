import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H]
    base = b * S * H + s * H

    # First pass: sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF
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

    # piecewise selection
    if p_val <= p_low:
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val >= p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    else:
        q = p_val - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        nd = poly * q / den

    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold (fp32)
    thr = mean + std * z

    # Apply y = max(x - thr, 0) along H dimension, store as bfloat16
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation:
          - Compute per (b, s) mean and std along H (fp32).
          - Compute z = inverse-normal CDF(target_sparsity) via A&S approximation in Triton.
          - Apply y = max(x - (mean + std*z), 0) and return in bfloat16.
        Host code only allocates tensors, sets up grid, and launches Triton kernels.
        """
        # Trivial case: no sparsity requested
        if target_sparsity == 0.0:
            return input_tensor.to(torch.bfloat16)

        # Ensure CUDA and contiguous input for Triton kernels
        B, S, H = input_tensor.shape
        in_fp32 = input_tensor.contiguous().to(torch.float32)
        device = in_fp32.device

        # Allocate per-row mean and std (fp32) buffers
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch compute_mean_std_fp32 kernel: one program per (b, s) row
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](in_fp32, mean_buf, std_buf, B, S, H, BLOCK=1024, num_warps=4)

        # Compute scalar z = icdf(target_sparsity) on device (fp32 tensor of size 1)
        z_tensor = torch.empty(1, dtype=torch.float32, device=device)
        p_val = float(target_sparsity)
        compute_icdf_scalar[(1,)](z_tensor, p_val, 0.02425, 1.0 - 0.02425)

        # Allocate output (bfloat16)
        out_bf16 = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        # Launch apply kernel: one program per (b, s) row
        apply_threshold_relu_to_bf16[grid_stats](in_fp32, out_bf16, mean_buf, std_buf, B, S, H, z_tensor, BLOCK=2048, num_warps=8)

        return out_bf16


def run(*args):
    return ModelNew()(*args)
