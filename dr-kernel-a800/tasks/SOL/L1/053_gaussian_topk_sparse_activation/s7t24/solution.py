import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row: pid in [0, B*S)
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

    # Pass 2: compute mean and std (population, unbiased=False)
    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_ptr, p_low: tl.float32, p_high: tl.float32, BLOCK: tl.constexpr):
    # Read p_val from device tensor p_ptr (shape [1])
    p_val = tl.load(p_ptr)  # scalar float32

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

    # piecewise computation for standard normal inverse CDF
    # lower tail
    q_low = tl.sqrt(-2.0 * tl.log(p_val))
    poly_low = -(((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    nd_low = poly_low / den_low

    # central region
    q_center = p_val - 0.5
    r = q_center * q_center
    poly_center = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    den_center = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
    nd_center = poly_center * q_center / den_center

    # upper tail
    q_upper = tl.sqrt(-2.0 * tl.log(1.0 - p_val))
    poly_upper = -(((((c1 * q_upper + c2) * q_upper + c3) * q_upper + c4) * q_upper + c5) * q_upper + c6)
    den_upper = (((((d1 * q_upper + d2) * q_upper + d3) * q_upper + d4) * q_upper + 1.0))
    nd_upper = poly_upper / den_upper

    # Select branch
    if p_val <= 0.02425:
        nd = nd_low
    elif p_val >= (1.0 - 0.02425):
        nd = nd_upper
    else:
        nd = nd_center

    # Store result to out_ptr[0] (1-element fp32 tensor)
    tl.store(out_ptr, nd)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (fp32) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    # Process H dimension in chunks
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle no sparsity
        if target_sparsity == 0.0:
            # return input as-is (sparse_output is all ones through ReLU of 0)
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and compute in fp32 for stats
        in_fp32 = inputs.to(torch.float32).contiguous()
        B, S, H = in_fp32.shape

        # Allocate mean and std buffers (fp32, [B*S])
        mean_ptr = torch.empty(B * S, dtype=torch.float32, device=in_fp32.device)
        std_ptr = torch.empty(B * S, dtype=torch.float32, device=in_fp32.device)

        # Launch compute_mean_std_fp32 kernel
        BLOCK = 2048
        grid = (B * S,)
        compute_mean_std_fp32[grid](in_fp32, mean_ptr, std_ptr, B, S, H, BLOCK)

        # Prepare p tensor (device scalar)
        p_tensor = torch.tensor([target_sparsity], dtype=torch.float32, device=in_fp32.device)

        # Allocate output y (fp32) and z_out (fp32 scalar)
        out_fp32 = torch.empty_like(in_fp32)  # temporary fp32 output for computation
        z_out = torch.empty(1, dtype=torch.float32, device=in_fp32.device)  # 1-element tensor for z

        # Launch compute_icdf_scalar kernel: out = icdf(p)
        p_low = 0.02425
        p_high = 1.0 - p_low
        compute_icdf_scalar[(1,)](z_out, p_tensor, p_low, p_high, BLOCK)

        # Launch apply_threshold_relu_to_bf16 kernel
        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](
            in_fp32, out_fp32, mean_ptr, std_ptr, z_out, B, S, H, BLOCK
        )

        # Convert to bfloat16 for final output
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
