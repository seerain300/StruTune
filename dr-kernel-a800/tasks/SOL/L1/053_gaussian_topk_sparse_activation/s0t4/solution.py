import math
import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean across the last dimension (F), one program per row
@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F] contiguous along last dim
    mean_out_ptr,      # *fp32, shape [NROWS] where NROWS = B*S
    F,                 # int32 (feature dimension)
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_base = pid * F

    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean = sum_val / F
    tl.store(mean_out_ptr + pid, mean)


# Kernel: compute sum of squares per row across the last dimension (F), one program per row
@triton.jit
def sumsq_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    sumsq_out_ptr,     # *fp32, shape [NROWS]
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_base = pid * F

    sum_sq = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    tl.store(sumsq_out_ptr + pid, sum_sq)


# Kernel: apply cutoff = mean + std * z and y = max(0, x - cutoff) per row
@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    mean_ptr,          # *fp32, shape [NROWS]
    std_ptr,           # *fp32, shape [NROWS]
    z,                 # fp32 scalar
    outputs_ptr,       # *fp32, shape [B, S, F]
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_base = pid * F
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * z

    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(outputs_ptr + row_base + f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous; compute in fp32
        assert inputs.is_cuda, "Input must be on CUDA for Triton kernels"
        B, S, F = inputs.shape
        NROWS = B * S
        inputs_f32 = inputs.contiguous().to(torch.float32)

        # Allocate per-row stats
        mean = torch.empty(NROWS, device=inputs.device, dtype=torch.float32)
        sumsq = torch.empty(NROWS, device=inputs.device, dtype=torch.float32)

        # Launch kernels for mean and sum of squares
        BLOCK_F = 1024  # tuneable block size; 1024 works well for large F
        mean_lastdim_kernel[(NROWS,)](
            inputs_f32, mean, F, BLOCK_F, num_warps=4
        )
        sumsq_lastdim_kernel[(NROWS,)](
            inputs_f32, sumsq, F, BLOCK_F, num_warps=4
        )

        # Compute std = sqrt(E[x^2] - (E[x])^2)
        mean_per_row = mean.view(B, S)           # [B, S]
        sumsq_per_row = sumsq.view(B, S)         # [B, S]
        std = torch.sqrt(sumsq_per_row / F - mean_per_row * mean_per_row)  # [B, S], fp32

        # Compute z using torch (host-side) from target sparsity:
        # z = norm.ppf(target_sparsity) = erfinv(2*sparsity - 1) * sqrt(2)
        z_scalar = float(torch.special.erfinv(2.0 * target_sparsity - 1.0).item()) * math.sqrt(2.0)

        # Prepare mean and std as [NROWS] for apply kernel
        mean_flat = mean                                  # [NROWS]
        std_flat = std.view(NROWS).contiguous()         # [NROWS]

        # Allocate output
        outputs = torch.empty_like(inputs_f32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[(NROWS,)](
            inputs_f32, mean_flat, std_flat, z_scalar, outputs, F, BLOCK_F, num_warps=4
        )

        # Cast back to original dtype
        return outputs.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
