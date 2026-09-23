import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per row: row id in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Base offset for this row in flattened [B, S, H]
    base = b * S * H + s * H

    # First pass: compute sum and sum of squares in fp32
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
    var = sum_sq / n - mean * mean  # population variance, unbiased=False
    std = tl.sqrt(var)

    # Store mean and std (fp32) per row
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar z (icdf) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32 scalar

    # Base offset for this row in flattened layout
    base = b * S * H + s * H

    # Apply: y = max(x - thr, 0) in fp32, write as bfloat16
    offsets = tl.arange(0, BLOCK)
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
        # Ensure inputs have expected shape
        assert inputs.ndim == 3, "Input must be [batch_size, seq_len, intermediate_size]"
        B, S, H = inputs.shape

        # If no sparsity requested, return inputs cast to bfloat16
        if target_sparsity <= 0.0:
            return inputs.to(torch.bfloat16)

        # Allocate per-row stats buffers (fp32) on device
        mean = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        std = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)

        # Kernel launch to compute mean and std
        BLOCK_STATS = 2048
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](
            inputs, mean, std, B, S, H, BLOCK=BLOCK_STATS, num_warps=4
        )

        # Compute inverse normal CDF z using PyTorch (standard normal): z = erfinv(2p - 1)
        # Note: target_sparsity is in (0, 1)
        z_scalar = torch.erfinv(2.0 * target_sparsity - 1.0).to(torch.float32).to(inputs.device)
        z_buf = z_scalar.view(1)  # 1-element tensor on device

        # Allocate output in bfloat16
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=inputs.device)

        # Apply threshold + ReLU in Triton
        grid_apply = (B * S,)
        apply_threshold_relu_to_bf16[grid_apply](
            inputs, out, mean, std, B, S, H, z_buf, BLOCK=2048, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
