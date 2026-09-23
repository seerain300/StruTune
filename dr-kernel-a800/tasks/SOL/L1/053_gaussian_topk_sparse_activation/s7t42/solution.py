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

    # First pass: accumulate sum and sum of squares in fp32 via scalar loads to minimize rounding error
    sum_val = 0.0
    sum_sq = 0.0
    i = 0
    while i < H:
        x = tl.load(in_ptr + base + i)
        x32 = x.to(tl.float32)
        sum_val += x32
        sum_sq += x32 * x32
        i += 1

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def icdf_and_apply(in_ptr, mean_ptr, std_ptr, out_ptr, B, S, H, p_val, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)

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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Compute inverse normal CDF z via A&S approximation
    # piecewise
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

    z = nd  # fp32

    # Compute threshold
    thr = mean + std * z  # fp32

    # Second pass: apply y = max(x - thr, 0.0) to the whole H
    base_row = b * S * H + s * H
    i = 0
    while i < H:
        x = tl.load(in_ptr + base_row + i)
        x32 = x.to(tl.float32)  # x is fp32; cast just in case
        y = x32 - thr
        y = tl.maximum(y, 0.0)  # relu
        tl.store(out_ptr + base_row + i, y)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs directly
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        inputs_fp32 = inputs.contiguous().to(torch.float32)  # compute in fp32 to match PyTorch's default behavior

        B = inputs_fp32.shape[0]
        S = inputs_fp32.shape[1]
        H = inputs_fp32.shape[2]

        # Allocate fp32 buffers for mean and std per row
        mean = torch.empty((B * S,), dtype=torch.float32, device=inputs_fp32.device)
        std = torch.empty((B * S,), dtype=torch.float32, device=inputs_fp32.device)

        # Launch compute_mean_std_fp32 kernel: one program per row
        grid = (B * S,)
        # Choose a moderate BLOCK; since we're doing scalar per-iteration loads, BLOCK doesn't matter here
        compute_mean_std_fp32[grid](inputs_fp32, mean, std, B, S, H, BLOCK=1)

        # Prepare output tensor in fp32 for computation, then cast to bfloat16 at the end
        out_fp32 = torch.empty((B, S, H), dtype=torch.float32, device=inputs_fp32.device)

        # Launch icdf_and_apply kernel
        # Pass target_sparsity as float
        grid = (B * S,)
        icdf_and_apply[grid](inputs_fp32, mean, std, out_fp32, B, S, H, target_sparsity, BLOCK=1)

        # Cast output to bfloat16 to match original model's return type
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
