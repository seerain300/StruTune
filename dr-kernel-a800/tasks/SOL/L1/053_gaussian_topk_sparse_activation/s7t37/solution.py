import torch
import triton
import triton.language as tl


@triton.jit
def compute_sum_sumsq_fp32(in_ptr, sum_ptr, sumsq_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H

    total_sum = 0.0
    total_sumsq = 0.0

    i = 0
    while i < H:
        # Scalar accumulation to avoid reduction/mask issues
        x = tl.load(in_ptr + base + i).to(tl.float32)
        total_sum += x
        total_sumsq += x * x
        i += 1

    tl.store(sum_ptr + row, total_sum)
    tl.store(sumsq_ptr + row, total_sumsq)


@triton.jit
def compute_mean_std_fp32(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)

    sum_val = tl.load(sum_ptr + row)
    sum_sq = tl.load(sumsq_ptr + row)

    mean = sum_val / H
    var = sum_sq / H - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        BLOCK: tl.constexpr):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF.
    # piecewise:
    # if p <= p_low: z = sqrt(-2*log(p))
    # elif p >= p_high: z = sqrt(-2*log(1-p))
    # else: centered polynomial with p-0.5
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
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    base = b * S * H + s * H
    # Iterate over H, compute y = max(x - thr, 0) and store as bfloat16
    i = 0
    while i < H:
        x = tl.load(in_ptr + base + i).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + i, y.to(tl.bfloat16))
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure we operate on CUDA tensors and use bfloat16
        assert inputs.is_cuda, "inputs must be on CUDA device"
        assert inputs.dtype == torch.bfloat16, "inputs must be bfloat16"

        # If no sparsity requested, return inputs directly (dtype unchanged)
        if target_sparsity == 0.0:
            return inputs

        B = inputs.shape[0]
        S = inputs.shape[1]
        H = inputs.shape[2]

        # Allocate buffers for sums, sumsq, mean, std, and z
        sum_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        z_buf = torch.empty((), dtype=torch.float32, device=inputs.device)

        # Launch kernel to compute sum and sumsq (fp32)
        grid = (B * S,)
        compute_sum_sumsq_fp32[grid](inputs, sum_buf, sumsq_buf, B, S, H, BLOCK=1024, num_warps=4)

        # Compute mean and std (fp32) per row
        compute_mean_std_fp32[grid](sum_buf, sumsq_buf, mean_buf, std_buf, B, S, H, BLOCK=1024, num_warps=4)

        # Compute icdf scalar z for target_sparsity
        p_val = float(target_sparsity)
        p_low = 0.02425
        p_high = 1.0 - p_low

        # A&S constants
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00

        compute_icdf_scalar[(1,)](z_buf, p_val, p_low, p_high,
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4,
                                  BLOCK=1, num_warps=1)

        # Prepare output tensor (bfloat16), same shape and strides as input
        out = torch.empty_like(inputs)

        # Launch apply kernel: per-row thresholding and relu
        apply_threshold_relu_to_bf16[grid](inputs, out, mean_buf, std_buf, z_buf, B, S, H, BLOCK=1024, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
