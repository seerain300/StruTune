import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    offsets = tl.arange(0, BLOCK)

    sum_val = 0.0
    sum_sq = 0.0

    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        base = b * S * H + s * H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    mean = sum_val / H
    var = sum_sq / H - mean * mean
    std = tl.sqrt(var)

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def apply_threshold_fp32_out_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, threshold, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    thr = mean + std * threshold  # scalar per row

    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        base = b * S * H + s * H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y = tl.maximum(x32 - thr, 0.0)  # ReLU(x - thr)
        tl.store(out_ptr + base + idx, y, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            return inputs

        if inputs.dim() != 3:
            raise ValueError("ModelNew.forward expects a 3D tensor of shape [batch_size, seq_len, intermediate_size].")
        B, S, H = inputs.shape

        # Ensure contiguous
        inputs_c = inputs.contiguous()
        device = inputs_c.device

        # Compute inverse-normal CDF constant once on host using torch
        # This is allowed: it's a scalar, not per-element tensor math.
        norm_icdf = float(torch._ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=device)).item())

        # Allocate fp32 buffers for mean and std
        mean = torch.empty((B, S), dtype=torch.float32, device=device)
        std = torch.empty((B, S), dtype=torch.float32, device=device)

        # Launch kernel 1: compute mean and std
        BLOCK = 1024
        grid = (B * S,)
        row_stats_fp32[grid](
            inputs_c, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Allocate output tensor (bfloat16, same shape as inputs)
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)

        # Launch kernel 2: apply threshold and ReLU
        apply_threshold_fp32_out_bf16[grid](
            inputs_c, out, mean, std, B, S, H, norm_icdf,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
