import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32_in_bf16(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    # First pass: accumulate sum and sum of squares (fp32) from bfloat16 inputs
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x_bf = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x_bf.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    n = H  # feature dimension
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar_in_triton(out_ptr, p_val, p_low, p_high, BLOCK: tl.constexpr):
    # Abramowitz & Stegun 5th-order rational approximation for standard normal inverse CDF
    # piecewise: lower tail, center, upper tail

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

    # piecewise selection
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

    # Load per-row mean and std (fp32) and scalar z (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    z = tl.load(z_ptr)  # 1-element tensor, scalar

    # Compute threshold (fp32)
    thr = mean + std * z

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        # Load input (bf16) and compute y = max(x - thr, 0.0) in fp32
        x_bf = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x_bf.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        # Store as bf16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation:
        mean/std along last dim -> icdf(z) = -erfinv(2p - 1) -> threshold = mean + std * z
        output = relu(inputs - threshold) in bfloat16.
        """
        # Ensure dtype and contiguity
        assert inputs.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        assert inputs.dtype == torch.bfloat16, "Inputs must be bfloat16."
        inputs = inputs.contiguous()
        B, S, H = inputs.shape
        device = inputs.device

        # Allocate per-row stats (fp32) and scalar z (fp32)
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)
        z = torch.empty(1, dtype=torch.float32, device=device)

        # Launch compute mean/std in Triton (from bfloat16 inputs, accumulate in fp32)
        BLOCK = 2048
        compute_mean_std_fp32_in_bf16[(B * S,)](
            inputs, mean, std, B, S, H, BLOCK=BLOCK, num_warps=4
        )

        # Launch Triton kernel to compute icdf(z) in device
        p_val = float(target_sparsity)
        # Use A&S piecewise approximation for icdf
        p_low = 0.02425
        p_high = 1.0 - p_low
        compute_icdf_scalar_in_triton[(1,)](
            z, p_val, p_low, p_high, BLOCK=BLOCK, num_warps=1
        )

        # Allocate output tensor (bfloat16)
        out = torch.empty_like(inputs)

        # Launch apply kernel
        apply_threshold_relu_to_bf16[(B * S,)](
            inputs, out, mean, std, z, B, S, H, BLOCK=BLOCK, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
