import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_base = (b * S + s) * D  # offset into flattened [B*S, D]
    sum_val = 0.0
    sumsq_val = 0.0

    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D,         # int32
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load mean and std for this (b, s) row
    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z_score

    # Load input, apply activation
    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are contiguous
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        # Work in float32 for statistics, return in bfloat16
        B, S, D = inputs.shape
        device = inputs.device

        # Flatten rows for reduction
        X = inputs.view(B * S, D)

        # Allocate device buffers for sum/sumsq/mean/std
        sum_buf = torch.empty(B * S, device=device, dtype=torch.float32)
        sumsq_buf = torch.empty(B * S, device=device, dtype=torch.float32)
        mean_buf = torch.empty(B * S, device=device, dtype=torch.float32)
        std_buf = torch.empty(B * S, device=device, dtype=torch.float32)

        # Launch reduction kernel
        BLOCK_SIZE = 1024
        grid_reduce = (B * S,)
        reduce_mean_std_kernel[grid_reduce](
            X, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )

        # Compute mean and std in Triton
        grid_mean = (B * S,)
        compute_mean_std_kernel[grid_mean](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D, num_warps=1
        )

        # Compute z_score using PyTorch's exact inverse-normal on device
        sparsity = 1.0 - float(target_sparsity)  # original uses norm.icdf(target_sparsity) = -ndtri(1 - p)
        z_score_tensor = torch.tensor(sparsity, device=device, dtype=torch.float32)  # scalar device tensor

        # Allocate output (float32 for computation, cast to bfloat16 at end)
        out_f32 = torch.empty(B * S * D, device=device, dtype=torch.float32)

        # Launch apply activation kernel
        grid_apply = (B, S, triton.cdiv(D, BLOCK_SIZE))
        apply_activation_kernel[grid_apply](
            X, mean_buf, std_buf, z_score_tensor, out_f32, B, S, D, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )

        # Reshape and cast to bfloat16 to match original function's output dtype
        out = out_f32.view(B, S, D).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
