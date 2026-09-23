import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row; pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H] memory layout
    base = b * S * H + s * H

    # Pass 1: accumulate sum and sum of squares in fp32
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

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar(out_ptr, p_val, BLOCK: tl.constexpr):
    # A&S 5.2.23 center formula for standard normal inverse CDF: z = - (num*q)/denom
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

    q = p_val - 0.5
    r = q * q
    numerator = ((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6
    denominator = ((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
    z = - (numerator * q) / denominator  # negative sign per A&S 5.2.23

    # store the scalar icdf result (fp32) at out_ptr[0]
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr_bf16, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar icdf(z) from z_ptr[0] (fp32)
    z = tl.load(z_ptr)

    # Compute threshold in fp32
    thr = mean + std * z  # fp32 scalar

    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Second pass: apply y = max(x - thr, 0) in fp32, then cast to bfloat16 and store
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = tl.maximum(x - thr, 0.0)  # fp32
        # Cast to bfloat16 before store
        y_bf16 = y.to(tl.bfloat16)
        tl.store(out_ptr_bf16 + base + idx, y_bf16, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return inputs directly (no computation needed).
        if target_sparsity == 0.0:
            return inputs

        # Ensure inputs are contiguous: shape [B, S, H] with H = intermediate_size
        B, S, H = inputs.shape

        # Work in fp32 for stats and computation
        in_fp32 = inputs.to(torch.float32)

        # Allocate mean and std buffers (fp32) per row
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)

        # Launch compute_mean_std_fp32 kernel
        grid_stats = (B * S,)
        triton.run(compute_mean_std_fp32, grid=grid_stats, num_warps=4, num_stages=2, args=(in_fp32, mean_buf, std_buf, B, S, H, 2048))

        # Compute icdf scalar for target_sparsity (store in 1-element fp32 tensor on device)
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        triton.run(compute_icdf_scalar, grid=(1,), num_warps=1, num_stages=1, args=(z_buf, float(target_sparsity), 256))

        # Allocate output buffer as bfloat16 to match original return type
        out_bf16 = torch.empty_like(inputs, dtype=torch.bfloat16, device=inputs.device)

        # Launch apply kernel: write bfloat16 directly
        grid_apply = (B * S,)
        triton.run(apply_threshold_relu_to_bf16, grid=grid_apply, num_warps=4, num_stages=2, args=(in_fp32, out_bf16, mean_buf, std_buf, z_buf, B, S, H, 2048))

        # Return bfloat16 result
        return out_bf16


def run(*args):
    return ModelNew()(*args)
