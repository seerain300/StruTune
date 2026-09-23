import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Accumulate sum and sum of squares using BLOCKed reduction over H
    sum_val = 0.0
    sum_sq = 0.0

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)

    for j in range(0, H, BLOCK):
        idx = j + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = H  # population count
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar_bipartite(out_ptr, p_val, p_low, p_high):
    # Inverse standard normal CDF via Abramowitz & Stegun 5th-order rational approximation.
    # p_val: float in (0,1). out_ptr: 1-element fp32 tensor to store nd.
    # Central region (typical for sparsity like 0.1, 0.5, 0.9):
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

    # q = p - 0.5
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

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)

    # Apply y = max(x - thr, 0) in fp32, store as bfloat16
    for j in range(0, H, BLOCK):
        idx = j + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs unchanged
        if target_sparsity == 0.0 or inputs.shape[-1] == 0:
            return inputs

        # Ensure contiguous memory layout
        inputs = inputs.contiguous()

        # Expect 3D input: [B, S, H]
        assert inputs.ndim == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, H = inputs.shape

        device = inputs.device

        # Allocate fp32 buffers for mean and std of shape [B*S]
        mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Launch kernel to compute per-row mean and std
        BLOCK = 1024
        grid = (B * S,)
        compute_mean_std_fp32[grid](
            inputs, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute icdf(z) for target_sparsity using Triton kernel. Pass scalar p_val.
        z_out = torch.empty((1,), dtype=torch.float32, device=device)
        p_low = 0.02425
        p_high = 1.0 - p_low
        # Pass target_sparsity as Python float
        compute_icdf_scalar_bipartite[(1,)](
            z_out, target_sparsity, p_low, p_high,
            num_warps=1,
        )
        # z is stored as z_out[0]
        z_scalar = z_out[0]

        # Allocate output tensor [B, S, H] as bfloat16
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        # Launch apply threshold ReLU kernel
        apply_threshold_relu_to_bf16[grid](
            inputs, out, mean, std, B, S, H, z_out,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
