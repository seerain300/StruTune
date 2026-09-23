import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    # Pass 1: accumulate sum and sum of squares (fp32)
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

    # Store per-row mean and std (fp32 scalars)
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar_triton(out_ptr, p_val, p_low=0.02425, p_high=0.97575, BLOCK: tl.constexpr=1):
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

    # piecewise logic
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

    # Write y = max(x - thr, 0.0) in bfloat16
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        # x is fp32 here (in_ptr is fp32 tensor from host); compute in fp32
        diff = x - thr
        y = tl.maximum(diff, 0.0)  # fp32
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Only do data preparation and kernel launches in host; no tensor math.
        # Ensure inputs are 3D: [B, S, H]
        assert inputs.dim() == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        B, S, H = inputs.shape

        # Work in float32 for statistics and computation
        in_tensor_f32 = inputs.contiguous().to(torch.float32)
        device = in_tensor_f32.device

        # Allocate outputs (final bfloat16)
        out_tensor = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        # Allocate stats buffers (fp32): one element per (b, s) row
        mean_ptr = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_ptr = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Launch Triton compute_mean_std_fp32 kernel
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](in_tensor_f32, mean_ptr, std_ptr, B, S, H, BLOCK=1024, num_warps=4)

        # Launch Triton compute_icdf_scalar_triton kernel to get z as fp32 scalar in z_ptr
        z_ptr = torch.empty((), dtype=torch.float32, device=device)  # 0-dim tensor
        # Pass p_val as float32 (host scalar), Triton handles conversion
        compute_icdf_scalar_triton[(1,)](z_ptr, float(target_sparsity), BLOCK=1)

        # Launch Triton apply kernel to compute y = max(x - (mean + std*z), 0) and store bfloat16
        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](in_tensor_f32, out_tensor, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK=4096, num_warps=8)

        return out_tensor


def run(*args):
    return ModelNew()(*args)
