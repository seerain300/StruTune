import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row, pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H  # since tensor is [B, S, H], contiguous => b*S*H + s*H

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

    # Compute mean and std (population, unbiased=False)
    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Store per-row mean and std
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

    # piecewise selection based on p_val
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

    # Load per-row mean and std (fp32). mean_ptr/std_ptr are [B*S]
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    # offsets vector for a block
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Second pass: apply y = max(x - thr, 0) in fp32, store as bfloat16
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)  # input is fp32
        diff = x - thr
        y = tl.maximum(diff, 0.0)
        # Cast to bfloat16 and store
        y_bf16 = y.to(tl.bfloat16)
        tl.store(out_ptr + base + idx, y_bf16, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # If no sparsity requested, just return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and convert to fp32 for computation
        in_tensor_f32 = inputs.to(torch.float32).contiguous()
        B = in_tensor_f32.size(0)
        S = in_tensor_f32.size(1)
        H = in_tensor_f32.size(2)

        # Allocate per-row mean and std (fp32) on device
        mean_ptr = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_ptr = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch mean/std kernel
        BLOCK = 1024
        compute_mean_std_fp32[(B * S,)](
            in_tensor_f32, mean_ptr, std_ptr, B, S, H, BLOCK,
            num_warps=4, num_stages=2
        )

        # Prepare output tensor in bfloat16
        out_bf16 = torch.empty_like(inputs)

        # Launch icdf kernel: compute z = icdf(target_sparsity) as fp32 scalar on device
        z_f32 = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_val = float(target_sparsity)
        compute_icdf_scalar_triton[(1,)](z_f32, p_val, BLOCK=1)

        # Launch apply kernel: apply threshold per (b, s) row
        BLOCK_APPLY = 4096
        apply_threshold_relu_to_bf16[(B * S,)](
            in_tensor_f32, out_bf16, mean_ptr, std_ptr, z_f32, B, S, H, BLOCK_APPLY,
            num_warps=8, num_stages=2
        )

        return out_bf16


def run(*args):
    return ModelNew()(*args)
