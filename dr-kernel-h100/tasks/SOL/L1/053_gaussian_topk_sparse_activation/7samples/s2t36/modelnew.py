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
        # Each row is of length N; pointer offset = row_id * N + offs
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
    Triton elementwise gating:
    For each row, compute y = max(0, x - (mean + std * inv_cdf))
    x_ptr: [rows, N]
    mean_ptr/std_ptr: [rows]
    inv_cdf_ptr: [1] (scalar)
    out_ptr: [rows, N] output
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    start = tile_id * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load per-row stats
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    inv_cdf = tl.load(inv_cdf_ptr)  # scalar

    # Load input row tile
    x = tl.load(x_ptr + row_id * N + offs, mask=mask, other=0.0)

    # Compute gating
    threshold = mean + std * inv_cdf
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(out_ptr + row_id * N + offs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation using Triton kernels.
    Computes adaptive sparsity threshold based on input statistics:
      mean + std * inv_norm_cdf(target_sparsity)
    Then y = max(0, x - threshold).
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs

    # Cast to float32 for numerically stable reduction and gating
    x = inputs
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Reshape to [rows, N] where rows = B * S
    B, S, N = x.shape
    rows = B * S
    x_2d = x.view(rows, N)

    # 1) Triton reduction: per-row mean and population std (unbiased=False)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    std = torch.empty(rows, device=x.device, dtype=torch.float32)
    BLOCK_SIZE_RS = 1024
    reduce_mean_std_2d[(rows,)](x_2d, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_RS, num_warps=4)

    # 2) Compute inverse normal CDF for target_sparsity using torch (robust and fast)
    # inv_norm_cdf(p) ≈ sqrt(2) * erfinv(2p - 1)
    # Use torch.special.erfinv on a 1-element tensor on the same device
    p = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
    inv_cdf = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.special.erfinv(2.0 * p - 1.0)
    inv_cdf = inv_cdf.view(1)  # pass as 1-element tensor to Triton

    # 3) Triton elementwise gating over 2D tiles
    out_2d = torch.empty((rows, N), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_GT = 1024
    num_tiles = (N + BLOCK_SIZE_GT - 1) // BLOCK_SIZE_GT
    grid = (rows, num_tiles)
    gate_rows_2d[grid](x_2d, mean, std, inv_cdf, out_2d, rows, N, BLOCK_SIZE=BLOCK_SIZE_GT, num_warps=4)

    # Reshape back to [B, S, N] and cast to bfloat16 to match original behavior
    out = out_2d.view(B, S, N).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape [batch_size, seq_len, intermediate_size]
        assert len(args) == 1, "ModelNew expects a single input tensor"
        return run(*args)