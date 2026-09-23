import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    # First pass: accumulate sum and sum of squares (fp32)
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

    n = H  # feature dimension
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std (fp32)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, p_low, p_high,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4, BLOCK: tl.constexpr):
    # Compute standard normal inverse CDF (quantile) for given p_val using A&S approximation
    # Piecewise: low, center, high
    if p_val <= p_low:
        # lower tail
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val >= p_high:
        # upper tail
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    else:
        # center region
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

    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Second pass: apply y = max(x - thr, 0) in fp32, store as bfloat16
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.where(y > 0.0, y, 0.0)  # relu
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of run(inputs, target_sparsity):
        - Compute per-row (b, s) mean and std (fp32).
        - Compute inverse normal CDF z for target_sparsity in Triton.
        - Apply y = max(x - (mean + std*z), 0) and return bfloat16.
        """
        # Handle trivial case
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA tensor and contiguous layout
        if not inputs.is_cuda:
            inputs = inputs.to('cuda')
        inputs = inputs.contiguous()
        B, S, H = inputs.shape

        # Allocate output buffer (bfloat16)
        out = torch.empty_like(inputs, dtype=torch.bfloat16)

        # Allocate per-row stats (fp32): [B, S]
        mean = torch.empty((B, S), dtype=torch.float32, device=inputs.device)
        std = torch.empty((B, S), dtype=torch.float32, device=inputs.device)

        # Launch Triton kernel to compute mean and std
        BLOCK = 2048  # tuned for large H; mask handles boundaries
        grid = (B * S,)
        compute_mean_std_fp32[grid](inputs, mean, std, B, S, H, BLOCK=BLOCK, num_warps=4)

        # Prepare constants for A&S approximation
        # Abramowitz & Stegun constants
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

        # Piecewise bounds
        p_low = 0.02425
        p_high = 1.0 - p_low

        # z buffer (1 element), computed in Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Launch Triton kernel to compute icdf for target_sparsity
        compute_icdf_scalar[(1,)](z_buf, float(target_sparsity), p_low, p_high,
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4,
                                  BLOCK=1024, num_warps=1)

        # Launch Triton kernel to apply threshold and ReLU
        apply_threshold_relu_to_bf16[grid](inputs, out, mean, std, z_buf, B, S, H, BLOCK=BLOCK, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
