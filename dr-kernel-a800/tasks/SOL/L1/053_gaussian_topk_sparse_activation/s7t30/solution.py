import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row; grid size is B*S
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base pointer for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    # Accumulate sum and sum of squares (fp32)
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std (fp32)
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

    # store the scalar inverse cdf value
    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr=4096):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)  # per (b, s)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)  # fp32 scalar

    # Compute threshold (fp32)
    thr = mean + std * z

    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Second pass: apply y = max(x - thr, 0.0) in fp32, store as bfloat16
    for start in range(0, H, BLOCK):
        idx = start + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr  # fp32
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        inputs: [batch_size, seq_len, intermediate_size], dtype bfloat16
        target_sparsity: float in [0, 1]
        returns: [batch_size, seq_len, intermediate_size], dtype bfloat16, sparsified via adaptive threshold and relu.
        """
        # If no sparsity requested, return inputs unchanged (original code handles this too)
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguity
        inputs = inputs.contiguous()

        B, S, H = inputs.shape

        # Allocate fp32 buffers for mean and std per (b, s) row
        mean = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        std = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)

        # 1) Triton compute mean and std in fp32
        grid_stats = (B * S,)
        # Use a BLOCK that balances occupancy and loop count; 4096 keeps few iterations for typical H (8K-12K)
        compute_mean_std_fp32[grid_stats](inputs.to(torch.float32), mean, std, B, S, H, BLOCK=4096, num_warps=8, num_stages=2)

        # 2) Triton compute icdf scalar z
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        compute_icdf_scalar_triton[(1,)](z_buf, float(target_sparsity), num_warps=1, num_stages=1)

        # 3) Triton apply thresholding and ReLU
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=inputs.device)
        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](inputs.to(torch.float32), out, mean, std, B, S, H, z_buf, BLOCK=4096, num_warps=8, num_stages=2)

        return out


def run(*args):
    return ModelNew()(*args)
