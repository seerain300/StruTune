import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_fp32(in_ptr, mean_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row, pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H
    sum_val = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Ensure masked lanes contribute 0
        x = tl.where(mask, x, 0.0)
        sum_val += tl.sum(x, axis=0)
        i += BLOCK

    mean = sum_val / H
    tl.store(mean_ptr + row, mean)


@triton.jit
def compute_sum_sq_fp32(in_ptr, sumsq_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row, pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x = tl.where(mask, x, 0.0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    tl.store(sumsq_ptr + row, sum_sq)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high, BLOCK: tl.constexpr):
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

    p_low = 0.02425
    p_high = 1.0 - p_low

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
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, sumsq_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and sum of squares (fp32)
    mean = tl.load(mean_ptr + row)
    sumsq = tl.load(sumsq_ptr + row)
    # Compute std (population, unbiased=False)
    std = tl.sqrt(sumsq / H - mean * mean)

    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    # Apply y = max(x - thr, 0) over H, store as bfloat16
    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation:
        - Compute per (batch, seq) row mean and sum of squares in fp32 over feature dim H (population stats).
        - Compute z = inverse-normal CDF(target_sparsity) as scalar (A&S approximation) in Triton, pass to device.
        - Compute std = sqrt(sumsq/H - mean^2).
        - Apply thresholding: y = max(input - (mean + std*z), 0) and return in bfloat16.
        """
        assert input_tensor.is_cuda, "Input must be a CUDA tensor for Triton kernels."
        # Ensure contiguous [B, S, H] layout
        in_bsf = input_tensor
        B = in_bsf.shape[0]
        S = in_bsf.shape[1]
        H = in_bsf.shape[2]

        # Allocate fp32 stats buffers: [B*S]
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=in_bsf.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=in_bsf.device)

        # Launch compute_mean_fp32 and compute_sum_sq_fp32 kernels
        grid = (B * S,)
        triton.run(compute_mean_fp32, grid, in_bsf, mean_buf, B, S, H, BLOCK=2048)
        triton.run(compute_sum_sq_fp32, grid, in_bsf, sumsq_buf, B, S, H, BLOCK=2048)

        # Allocate scalar output buffer for icdf and compute z on device
        z_buf = torch.empty(1, dtype=torch.float32, device=in_bsf.device)
        p_val = float(target_sparsity)  # pass as float, not tensor math
        triton.run(compute_icdf_scalar, grid, z_buf, p_val, 0.02425, 1.0 - 0.02425, BLOCK=1)

        # Prepare output tensor (bfloat16), matching input shape
        out = torch.empty_like(in_bsf, dtype=torch.bfloat16)

        # Launch apply kernel
        triton.run(apply_threshold_relu_to_bf16, grid, in_bsf, out, mean_buf, sumsq_buf, B, S, H, z_buf, BLOCK=2048)

        return out


def run(*args):
    return ModelNew()(*args)
