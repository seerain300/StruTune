import torch
import triton
import triton.language as tl


@triton.jit
def compute_stats_fp32(in_ptr, mean_ptr, std_ptr, B, S, H):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H  # since layout is [B, S, H] contiguous, this is correct

    # Accumulate sum and sum of squares in fp32 scalars
    sum_val = 0.0
    sum_sq = 0.0
    i = 0
    while i < H:
        x = tl.load(in_ptr + base + i)
        # Cast to fp32 for stable accumulation
        x = x.to(tl.float32)
        sum_val += x
        sum_sq += x * x
        i += 1

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4,
                        BLOCK: tl.constexpr):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF
    # Piecewise:
    # if p <= p_low: z = sqrt(-2*log(p))
    # elif p >= p_high: z = sqrt(-2*log(1-p))
    # else: centered polynomial with p-0.5
    # Store result to out_ptr[0] as fp32.
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

    # Compute threshold in fp32
    thr = mean + std * z  # fp32

    # Base offset for this row (layout [B, S, H] contiguous)
    base = b * S * H + s * H

    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr  # fp32
        # ReLU: y = max(y, 0)
        y = tl.maximum(y, 0.0)
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return input as bfloat16
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure inputs are on CUDA for Triton
        assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure contiguous [B, S, H] layout
        inputs = inputs.contiguous()

        B, S, H = inputs.shape

        # Allocate fp32 buffers for mean and std per (b, s)
        mean_buf = torch.empty((B, S), dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty((B, S), dtype=torch.float32, device=inputs.device)

        # Launch compute_stats_fp32 kernel: one program per (b, s) row
        grid_stats = (B * S,)
        compute_stats_fp32[grid_stats](inputs, mean_buf, std_buf, B, S, H)

        # Compute icdf scalar z for target_sparsity (fp32, 1-element tensor)
        z = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_val = float(target_sparsity)
        p_low = 0.02425
        p_high = 1.0 - p_low

        # A&S coefficients
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

        grid_icdf = (1,)
        compute_icdf_scalar[grid_icdf](z, p_val, p_low, 1.0 - p_low, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4)

        # Allocate output buffer (bfloat16)
        out = torch.empty_like(inputs, dtype=torch.bfloat16, device=inputs.device)

        # Launch apply thresholding kernel
        grid_apply = (B * S,)
        BLOCK = 2048  # good trade-off for large H; adjust warps accordingly
        apply_threshold_relu_to_bf16[grid_apply](inputs, out, mean_buf, std_buf, z, B, S, H, BLOCK)

        return out


def run(*args):
    return ModelNew()(*args)
