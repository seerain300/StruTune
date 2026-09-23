import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 scalars
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    base = (b * S + s) * D
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)
    tl.store(SUM_ptr + pid, total_sum)
    tl.store(SUMSQ_ptr + pid, total_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D,         # int32 scalars
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
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
    OUT_ptr,         # *bf16, output [B, S, D]
    B, S, D,         # int32 scalars
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # linear index for this row

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    y_bf16 = y.to(tl.bfloat16)
    tl.store(OUT_ptr + base_idx + offs, y_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float = 0.0):
        # Ensure CUDA and contiguous (no torch ops)
        if not inputs.is_cuda:
            inputs = inputs.to('cuda')
        inputs = inputs.contiguous()
        device = inputs.device
        B = inputs.shape[0]
        S = inputs.shape


def run(*args):
    return ModelNew()(*args)
