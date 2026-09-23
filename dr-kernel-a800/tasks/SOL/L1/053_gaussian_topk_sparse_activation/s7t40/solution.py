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
    base = b * S * H + s * H

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

    n = H  # feature dimension
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)

    # Load scalar z (icdf) from z_ptr[0] as fp32
    z = tl.load(z_ptr)

    # Compute threshold
    thr = mean + std * z  # fp32

    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H

    # Second pass: apply y = max(x - thr, 0) in fp32, store as bfloat16
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float = 0.0):
        # x: [B, S, H] in bfloat16
        assert x.dtype == torch.bfloat16, "Input must be bfloat16"
        B, S, H = x.shape

        # 1) Compute mean and std per (b, s) using Triton kernel (fp32 accumulators)
        in_fp32 = x.to(torch.float32)
        mean_buf = torch.empty((B, S), dtype=torch.float32, device=x.device)
        std_buf = torch.empty((B, S), dtype=torch.float32, device=x.device)

        BLOCK = 2048
        grid = (B * S,)
        compute_mean_std_fp32[grid](in_fp32, mean_buf, std_buf, B, S, H, BLOCK=BLOCK, num_warps=8)

        # 2) Compute icdf(target_sparsity) using PyTorch's special function for accuracy:
        # Standard normal: icdf(p) = sqrt(2) * erfinv(2p - 1)
        p = float(target_sparsity)
        if p <= 0.0 or p >= 1.0:
            # if sparsity is 0 or 1, return original input
            return x
        z = torch.special.erfinv(2.0 * p - 1.0)  # fp32 tensor of shape [1] on device
        z_buf = z  # 1-element fp32 tensor on device

        # 3) Apply thresholding and write output in bfloat16 using Triton
        out = torch.empty_like(x)
        apply_threshold_relu_to_bf16[grid](in_fp32, out, mean_buf, std_buf, z_buf, B, S, H, BLOCK=BLOCK, num_warps=8)

        return out


def run(*args):
    return ModelNew()(*args)
