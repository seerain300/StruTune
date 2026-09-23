import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S

    # Base linear offset for this row (contiguous layout: [B, S, D])
    base = b * S * D + s * D

    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over features in chunks
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Compute mean and std (fp32)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)  # clamp for numerical stability
    std = tl.sqrt(var)

    # Write per-row sum, sumsq, mean, std
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bf16, output [B, S, D]
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # linear index offset

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load mean and std for this (b, s) row
    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z_score

    # Load input, apply activation, store as bf16
    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold  # scalar broadcast
    y = tl.maximum(y, 0.0)  # ReLU
    y_bf16 = y.to(tl.bfloat16)
    tl.store(OUT_ptr + base_idx + offs, y_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Early return for no sparsity
        if target_sparsity == 0.0:
            # Return input as bfloat16 to match original behavior
            return x.to(torch.bfloat16)

        # Ensure contiguous and device placement
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        x = x.contiguous()

        # Dimensions
        B, S, D = x.shape

        # Allocate device buffers for sums and stats (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Device scalar for sparsity (avoid torch ops in host)
        p_buf = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=x.device)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)

        # Launch reduction + mean/std kernel: one program per (b, s)
        grid_reduce = (B * S,)
        reduce_mean_std_kernel[grid_reduce](
            x, sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D,
            BLOCK_SIZE=1024, num_warps=8,
        )

        # Launch ndtri approximation kernel: compute scalar z-score
        ndtri_approx_kernel[(1,)](
            p_buf, z_buf, num_warps=1,
        )

        # Prepare output tensor (bfloat16) and launch activation kernel
        out = torch.empty(B, S, D, dtype=torch.bfloat16, device=x.device)

        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            x, mean_buf, std_buf, z_buf, out,
            B, S, D,
            BLOCK_SIZE=1024, num_warps=8,
        )

        return out


def run(*args):
    return ModelNew()(*args)
