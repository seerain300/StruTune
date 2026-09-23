import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_2d(x_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton reduction: compute per-row mean and population std (unbiased=False) across last dim.
    x_ptr: pointer to input flattened as [rows, N] (row-major), contiguous
    mean_ptr/std_ptr: per-row outputs [rows], float32
    N: number of columns (int32)
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0

    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def gate_rows_2d(x_ptr, mean_ptr, std_ptr, inv_cdf_ptr, out_ptr, rows, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating: y = max(0, x - (mean + std * inv_cdf))
    Grid: (rows, num_tiles), each program handles one row and one tile of N.
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load inputs
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    cutoff = mean + std * inv_cdf
    gated = x - cutoff
    gated = tl.maximum(gated, 0.0)
    tl.store(out_ptr + row_id * N + offs, gated, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized version of the original run:
    1) Compute mean and std per row across feature dimension (last dim).
    2) Compute inverse normal CDF for target_sparsity on host (torch.special.erfinv).
    3) Apply gating y = max(0, x - (mean + std * inv_cdf)) via Triton kernel.
    """
    if target_sparsity == 0.0:
        return inputs

    # Ensure input is on CUDA and contiguous
    assert inputs.is_cuda, "Input must be on CUDA for Triton kernels"
    x = inputs.contiguous()
    # Compute in float32
    x_f32 = x.to(torch.float32)

    # Flatten to 2D [rows, N], where N = last dimension
    B, S, N = x_f32.shape
    rows = B * S
    x_2d = x_f32.view(rows, N)

    # 1) Per-row mean and std
    mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inv_norm_cdf(target_sparsity) on host using erfinv:
    # inv_norm_cdf(p) ≈ sqrt(2) * erfinv(2p - 1)
    p = torch.tensor(target_sparsity, dtype=torch.float32, device=x_f32.device)
    inv_cdf = torch.sqrt(torch.tensor(2.0, device=x_f32.device, dtype=torch.float32)) * torch.special.erfinv(
        2.0 * p - 1.0
    )

    # 3) Elementwise gating in Triton
    out_2d = torch.empty((rows, N), device=x_f32.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape [batch_size, seq_len, intermediate_size]
        assert len(args) == 1, "ModelNew expects a single input tensor"
        return run(*args)


def run(*args):
    return ModelNew()(*args)
