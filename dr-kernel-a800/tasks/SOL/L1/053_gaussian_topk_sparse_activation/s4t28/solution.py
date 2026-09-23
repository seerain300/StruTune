import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _sparsify_kernel(x_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification per row:
      cutoff = mean + std * std_multiplier
      out = max(x - cutoff, 0)
    x_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    std_multiplier: scalar f32 (host float passed; no torch ops on device tensors)
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)

    # Compute cutoff
    cutoff = mean + std * std_multiplier

    # Iterate over the row in chunks, apply ReLU after threshold
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes:
          mean = inputs.mean(dim=-1, keepdim=True)
          std = inputs.std(dim=-1, keepdim=True, unbiased=False)
          cutoff = mean + std * norm.icdf(target_sparsity)
          outputs = relu(inputs - cutoff)
        All per-row reductions and elementwise ops are done in Triton kernels.
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in f32
        x = inputs.contiguous()
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)

        # Shapes
        B, S, N = x32.shape
        total = B * S
        device = x32.device

        # Allocate stats buffers
        mean = torch.empty(total, dtype=torch.float32, device=device)
        std = torch.empty(total, dtype=torch.float32, device=device)

        # Launch reduction kernel (tuned for typical N up to 16k)
        BLOCK_SIZE_R = 2048
        grid = (total,)
        _row_reduce_mean_std_kernel[grid](x32, mean, std, N, BLOCK_SIZE_R, num_warps=8, num_stages=4)

        # Compute scalar std_multiplier using torch.erfinv for high accuracy:
        # norm.icdf(p) = sqrt(2) * erfinv(2p - 1)
        # We avoid creating device tensors in forward; do this on host as a Python float.
        # torch.erfinv expects input in [-1, 1]; for target_sparsity in (0,1), 2p-1 in (-1,1).
        std_multiplier = float(torch.erfinv(torch.tensor(2.0 * target_sparsity - 1.0)).item())

        # Allocate output
        out32 = torch.empty_like(x32)

        # Launch sparsification kernel
        BLOCK_SIZE_E = 2048
        grid2 = (total,)
        _sparsify_kernel[grid2](x32, mean, std, out32, N, std_multiplier, BLOCK_SIZE_E, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out32.to(orig_dtype)


def run(*args):
    return ModelNew()(*args)
