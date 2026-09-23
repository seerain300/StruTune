import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H

    # First pass: accumulate sum and sum of squares (fp32), using masks to exclude out-of-range lanes
    sum_val = 0.0
    sum_sq = 0.0

    i = 0
    while i < H:
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < H
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
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

    p_low_scalar = 0.02425
    p_high_scalar = 1.0 - p_low_scalar

    if p_val <= p_low_scalar:
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val >= p_high_scalar:
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
    # One program per (b, s) row
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

    # Second pass: apply y = max(x - thr, 0) over H, store as bfloat16
    base = b * S * H + s * H

    # Iterate over H in BLOCK-sized chunks, only operate on valid offsets
    i = 0
    while i < H:
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < H
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # clamp to relu
        # Store only valid offsets
        tl.store(out_ptr + base + offsets, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation:
        - Compute per (batch, seq) row mean and std in fp32 over feature dim H (population, unbiased=False).
        - Compute z = inverse-normal CDF(target_sparsity) as scalar on device (A&S approximation).
        - Apply thresholding: y = max(input - (mean + std*z), 0) and return in bfloat16.
        """
        assert input_tensor.is_cuda, "Input must be a CUDA tensor for Triton implementation."
        assert input_tensor.dtype == torch.bfloat16, "Input tensor must be bfloat16."
        B = input_tensor.size(0)
        S = input_tensor.size(1)
        H = input_tensor.size(2)

        # Ensure contiguous input for pointer arithmetic
        in_ptr = input_tensor.contiguous()
        out_ptr = torch.empty_like(in_ptr)  # output bfloat16

        # Buffers for per-row stats (fp32) with shape [B*S]
        mean_ptr = torch.empty(B * S, dtype=torch.float32, device=in_ptr.device)
        std_ptr = torch.empty(B * S, dtype=torch.float32, device=in_ptr.device)

        # Launch compute_mean_std_fp32 over grid (B*S,)
        compute_mean_std_fp32[(B * S,)](
            in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK=2048
        )

        # Compute z = icdf(target_sparsity) using A&S approximation, write to 1-element fp32 tensor
        z_buf = torch.empty(1, dtype=torch.float32, device=in_ptr.device)
        p_val = float(target_sparsity)
        compute_icdf_scalar[(1,)](z_buf, p_val, 0.02425, 0.97575, BLOCK=1)

        # Launch apply kernel: apply threshold relu per (b, s) row
        apply_threshold_relu_to_bf16[(B * S,)](
            in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_buf, BLOCK=2048
        )

        return out_ptr


def run(*args):
    return ModelNew()(*args)
