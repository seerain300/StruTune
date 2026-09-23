import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row; pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] layout
    base = b * S * H + s * H

    # Accumulate sum and sum of squares across H
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Reduce this vector chunk to scalars
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std (shape: [B*S], will be reshaped after)
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
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
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

    # Second pass: apply y = max(x - thr, 0) in fp32, store as bfloat16
    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # relu
        # cast to bfloat16 for output
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + base + idx, y, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation of:
          - compute mean and std across feature dim (last) with keepdim=True (population, unbiased=False)
          - compute inverse normal CDF of target_sparsity (scalar z) via A&S approximation in Triton
          - compute threshold = mean + std * z (per row)
          - output = relu(inputs - threshold) in bfloat16
        No tensor math in host code.
        """
        # Ensure inputs are 3D: [batch_size, seq_len, intermediate_size]
        assert inputs.dim() == 3, "inputs must be of shape [B, S, H]"
        B, S, H = inputs.shape

        # Make contiguous and use float32 for stats computation
        in_f32 = inputs.contiguous().to(torch.float32)

        # Allocate per-row stats buffers (fp32), shape [B*S] to index by row id
        means = torch.empty(B * S, dtype=torch.float32, device=in_f32.device)
        stds = torch.empty(B * S, dtype=torch.float32, device=in_f32.device)

        # Launch Triton kernel to compute mean and std per (b, s) row
        # Grid: one program per row
        grid = (B * S,)
        compute_mean_std_fp32[grid](in_f32, means, stds, B, S, H, BLOCK=4096)

        # Prepare mean/std tensors shaped [B, S, 1] for apply kernel
        # Convert to contiguous [B*S] for kernel to read per row
        # Apply kernel expects pointers to [B*S] vectors
        # Note: we will pass mean/std as 1D vectors and compute per row.

        # Launch Triton kernel to compute icdf scalar z for target_sparsity
        z_out = torch.empty(1, dtype=torch.float32, device=in_f32.device)
        compute_icdf_scalar_triton[(1,)](z_out, target_sparsity)

        # Allocate output in bfloat16
        out_bf16 = torch.empty((B, S, H), dtype=torch.bfloat16, device=in_f32.device)

        # Launch apply kernel: per-row thresholding and relu
        apply_threshold_relu_to_bf16[grid](in_f32, out_bf16, means, stds, B, S, H, z_out, BLOCK=4096, num_warps=8)

        return out_bf16


def run(*args):
    return ModelNew()(*args)
