import math
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
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One Triton program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold (fp32)
    thr = mean + std * z

    # Second pass: apply y = max(x - thr, 0) in fp32, store as bfloat16
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = tl.maximum(x - thr, 0.0)
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return as-is
        if target_sparsity == 0.0:
            return inputs

        # Ensure inputs are contiguous and in fp32 for compute stability
        inputs_f32 = inputs.contiguous().to(torch.float32)

        B, S, H = inputs_f32.shape

        # Allocate mean and std buffers (fp32) for each row
        mean_buf = torch.empty(B * S, device=inputs.device, dtype=torch.float32)
        std_buf = torch.empty(B * S, device=inputs.device, dtype=torch.float32)

        # Launch kernel to compute mean and std
        BLOCK = 1024  # balanced block size
        grid = (B * S,)
        compute_mean_std_fp32[grid](inputs_f32, mean_buf, std_buf, B, S, H, BLOCK=BLOCK, num_warps=4, num_stages=2)

        # Compute standard normal icdf (z) using torch.special.erfinv (reliable and accurate)
        # z = Phi^(-1)(target_sparsity) = erfinv(2 * p - 1) / sqrt(2)
        p_tensor = torch.tensor(float(target_sparsity), device=inputs.device, dtype=torch.float32)
        z_val = torch.special.erfinv(2.0 * p_tensor - 1.0) / math.sqrt(2.0)  # fp32 tensor on device
        z_tensor = z_val.view(1)  # 1-element device tensor

        # Allocate output tensor (bfloat16)
        out_bf16 = torch.empty(inputs_f32.shape, device=inputs.device, dtype=torch.bfloat16)

        # Launch apply kernel over rows
        apply_threshold_relu_to_bf16[grid](inputs_f32, out_bf16, mean_buf, std_buf, B, S, H, z_tensor, BLOCK=BLOCK, num_warps=4, num_stages=2)

        return out_bf16


def run(*args):
    return ModelNew()(*args)
