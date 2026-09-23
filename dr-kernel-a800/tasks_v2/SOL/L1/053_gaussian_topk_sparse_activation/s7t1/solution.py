import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per row: row id in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    offsets = tl.arange(0, BLOCK)
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: accumulate sum and sum of squares
    i = 0
    base = b * S * H + s * H
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    mean = sum_val / H
    # population variance: E[x^2] - (E[x])^2
    var = sum_sq / H - mean * mean
    std = tl.sqrt(var)

    # Store results (fp32)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p, p_low, p_high):
    # Compute inverse standard normal CDF for p (scalar 0-dim torch tensor) using Abramowitz & Stegun approximation.
    # Store result into out_ptr[0] as fp32.

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

    p_low_c = 0.02425
    p_high_c = 1.0 - p_low_c

    # Load p as float (Triton treats tensor argument as a scalar for this kernel)
    p_val = p

    if p_val < p_low_c:
        # Lower tail
        q = tl.sqrt(-2.0 * tl.log(p_val))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    elif p_val > p_high_c:
        # Upper tail
        q = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
    else:
        # Center region
        q = p_val - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        nd = poly * q / den

    # Store scalar result
    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per row
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
        y = tl.maximum(y, 0.0)
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Cache for icdf scalar: key = (device, dtype, target_sparsity) -> tensor([1], dtype=float32)
        self._icdf_cache = {}

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return as-is
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous
        inputs = inputs.contiguous()

        # Dimensions
        assert inputs.ndim == 3, "inputs must be 3D: [B, S, H]"
        B, S, H = inputs.shape

        device = inputs.device

        # 1) Allocate mean and std buffers (fp32)
        mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # 2) Launch kernel to compute per-row mean and std
        BLOCK = 1024
        grid = (B * S,)
        compute_mean_std_fp32[grid](
            inputs, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # 3) Compute or fetch icdf for target_sparsity using Triton kernel
        # Cache per (device, dtype, target_sparsity). Note: dtype here is not used in icdf computation.
        key = (device, inputs.dtype, target_sparsity)
        z_out = self._icdf_cache.get(key)
        if z_out is None:
            z_out = torch.empty((1,), dtype=torch.float32, device=device)
            p = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
            # Launch once and cache
            compute_icdf_scalar[(1,)](
                z_out, p, 0.02425, 1.0 - 0.02425,
                num_warps=1,
            )
            # Store in cache
            self._icdf_cache[key] = z_out

        # 4) Prepare output (bfloat16) and launch apply kernel
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        apply_threshold_relu_to_bf16[grid](
            inputs, out, mean, std, B, S, H, z_out,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
