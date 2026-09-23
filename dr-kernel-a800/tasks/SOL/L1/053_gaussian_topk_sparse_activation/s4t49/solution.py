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

    # Iterate over the row in chunks of BLOCK_SIZE
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        offset += BLOCK_SIZE

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _sparsify_rows_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification per row:
    out = relu(inputs - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    std_multiplier: scalar f32 computed by host from _ndtri(target_sparsity)
    """
    pid = tl.program_id(0)
    row_start = pid * N
    threshold = tl.load(mean_ptr + pid) + tl.load(std_ptr + pid) * std_multiplier

    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)
        offset += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized Gaussian-based top-k sparse activation.
        Computes adaptive sparsity threshold per (batch, seq) row:
        threshold = mean + std * norm.icdf(target_sparsity)
        Then outputs relu(inputs - threshold).
        All heavy computations (reduction and sparsification) are in Triton kernels.
        The std multiplier is computed on host using the original _ndtri to ensure exact match.
        """
        # Handle no sparsity
        if target_sparsity == 0.0:
            # Return a fresh tensor with same dtype and device
            return inputs.clone()

        # Ensure contiguous and compute in float32
        inputs_f32 = inputs.to(torch.float32).contiguous()
        B, S, N = inputs_f32.shape
        device = inputs_f32.device

        # Allocate per-row stats
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch reduction kernel
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inputs_f32, mean, std, N, BLOCK_SIZE=2048, num_warps=8, num_stages=4)

        # Compute std multiplier using original _ndtri to match exact behavior
        # Create scalar tensor on host with the same dtype as needed; _ndtri returns float32
        std_multiplier = float(_ndtri(torch.tensor(target_sparsity, dtype=torch.float32)))

        # Allocate output
        out_f32 = torch.empty_like(inputs_f32)

        # Launch elementwise sparsification kernel
        _sparsify_rows_kernel[grid](inputs_f32, mean, std, out_f32, N, std_multiplier, BLOCK_SIZE=4096, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
