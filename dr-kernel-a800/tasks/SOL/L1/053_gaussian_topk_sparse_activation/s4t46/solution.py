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
def _sparsify_row_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification for each (batch, seq) row:
      out = relu(inp - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int
    std_multiplier: f32 scalar
    """
    pid = tl.program_id(0)
    row_start_in = pid * N
    row_start_out = pid * N  # assuming out contiguous like input

    threshold = tl.load(mean_ptr + pid) + tl.load(std_ptr + pid) * std_multiplier

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start_in + idx, mask=mask, other=0.0)
        # ReLU of (x - threshold)
        diff = x - threshold
        y = tl.maximum(diff, 0.0)
        tl.store(out_ptr + row_start_out + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized top-k sparse activation:
        - Compute per-row mean and std across last dim.
        - Compute std multiplier via inverse normal CDF (A&S approximation) on host.
        - Apply ReLU(input - (mean + std * std_multiplier)) to produce sparse output.
        """
        # If no sparsity requested, return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

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

        # Compute std multiplier using host helper (no torch ops on device tensors in forward)
        std_multiplier = _ndtri(target_sparsity)  # host scalar

        # Allocate output
        out_f32 = torch.empty_like(inputs_f32)

        # Launch elementwise sparsification kernel
        _sparsify_row_kernel[grid](inputs_f32, mean, std, out_f32, N, std_multiplier, BLOCK_SIZE=4096, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
