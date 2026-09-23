import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Accumulate sum and sum of squares over H using BLOCKed reduction
    sum_val = 0.0
    sum_sq = 0.0

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)

    for j in range(0, H, BLOCK):
        idx = j + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = H  # population count
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_threshold_and_apply(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, p_val, p_low, p_high, BLOCK: tl.constexpr):
    # Single program computes inverse normal CDF of p_val and applies threshold per row.
    # Note: grid=(1,), so this runs once. We compute z and then process all rows sequentially.
    # This design allows us to keep Triton-only and avoid host-side tensor math.

    # Compute inverse normal CDF (standard) via Abramowitz & Stegun 5th-order rational approximation
    # Piecewise: lower tail, central region, upper tail.

    # Coefficients
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

    if p_val <= p_low:
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / den
    elif p_val >= p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / den
    else:
        q = p_val - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        z = poly * q / den

    # Now, process each row: pid in [0, B*S)
    for row in range(0, B * S):
        mean = tl.load(mean_ptr + row)
        std = tl.load(std_ptr + row)
        thr = mean + std * z  # fp32

        b = row // S
        s = row % S
        base = b * S * H + s * H

        # Apply threshold and store as bfloat16
        for j in range(0, H, BLOCK):
            idx = j + tl.arange(0, BLOCK)
            mask = idx < H
            x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            y = x - thr
            y = tl.maximum(y, 0.0)
            tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Trivial case: no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous
        inputs = inputs.contiguous()

        # Dimensions
        assert inputs.ndim == 3, "inputs must be 3D: [B, S, H]"
        B, S, H = inputs.shape
        device = inputs.device

        # Allocate mean and std buffers (fp32)
        mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # 1) Launch kernel to compute per-row mean and std (BLOCK=1024)
        compute_mean_std_fp32[(B * S,)](
            inputs, mean, std, B, S, H, BLOCK=1024, num_warps=4
        )

        # 2) Prepare output (bfloat16) and launch kernel that computes z and applies threshold
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        # Run a single-program kernel that computes z (from target_sparsity) and applies threshold
        # We pass target_sparsity as a Python float; Triton will treat it as a scalar.
        compute_threshold_and_apply[(1,)](
            inputs, out, mean, std, B, S, H, target_sparsity, 0.02425, 1.0 - 0.02425, BLOCK=1024, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
