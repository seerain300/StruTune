import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # First pass: accumulate sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

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
def compute_icdf_scalar_asgs(out_ptr, p_val, p_low, p_high, BLOCK: tl.constexpr):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF.
    # p_val: target_sparsity, Python float passed in.
    # p_low = 0.02425, p_high = 0.97575.

    # Coefficients (a-series)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    # Coefficients (b-series)
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Coefficients (c-series, used for lower and upper)
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    # Coefficients (d-series, used for lower and upper)
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Piecewise computation
    if p_val <= p_low:
        # lower tail: p < 0.02425
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val >= p_high:
        # upper tail: p > 0.97575
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    else:
        # central region
        q = p_val - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        nd = poly * q / den

    # Store scalar result
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
    # Load scalar z from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Apply y = max(x - thr, 0) (ReLU of (x - thr)) in fp32, store as bfloat16
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)  # assume in_ptr is fp32 input
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous
        inputs = inputs.contiguous()

        # Dimensions: [B, S, H]
        assert inputs.ndim == 3, "inputs must be 3D: [B, S, H]"
        B, S, H = inputs.shape
        device = inputs.device

        # 1) Compute mean and std per (b, s) using Triton kernel (population std, unbiased=False)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=device)

        in_f32 = inputs.to(torch.float32)
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](
            in_f32, mean_buf, std_buf, B, S, H, BLOCK=1024
        )

        # 2) Compute inverse normal CDF for target_sparsity using A&S Triton kernel (scalar)
        p_low = 0.02425
        p_high = 1.0 - p_low
        z_out = torch.empty(1, dtype=torch.float32, device=device)
        grid_icdf = (1,)
        compute_icdf_scalar_asgs[grid_icdf](z_out, target_sparsity, p_low, p_high, BLOCK=1)

        # 3) Apply threshold and ReLU to produce bfloat16 output
        out = torch.empty_like(inputs, dtype=torch.bfloat16)

        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](
            in_f32, out, mean_buf, std_buf, z_out, B, S, H, BLOCK=1024
        )

        return out


def run(*args):
    return ModelNew()(*args)
