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
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Uses a single polynomial in transformed variable t to avoid branching.
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load scalar p
    p = tl.load(p_dev_ptr)

    # Constants for A&S approximation
    p0 = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # Compute transformed variable
    x = 1.0 - p
    t = 1.0 / (1.0 + p0 * x)

    # Polynomial via Horner's method
    poly = a5 * t + a4
    poly = poly * t + a3
    poly = poly * t + a2
    poly = poly * t + a1
    poly = poly * t

    # z = sign(p - 0.5) * (1 / t_poly - t)
    z = tl.where(p < 0.5, -1.0, 1.0) * (1.0 / (t * poly) - t)

    # Store result
    tl.store(out_ptr, z)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(inp - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    std_multiplier: scalar f32 (std_norm for target sparsity)
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row mean and std
    mean_val = tl.load(mean_ptr + pid)
    std_val = tl.load(std_ptr + pid)
    threshold = mean_val + std_val * std_multiplier

    # Iterate over the row in chunks and apply ReLU
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation:
          - Compute per-row mean and std over last dim (float32).
          - Compute inverse standard normal CDF for target_sparsity (scalar) via Triton A&S.
          - Compute cutoff = mean + std * std_multiplier.
          - Apply out = relu(inputs - cutoff) in Triton, return in bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        x = inputs
        x_f32 = x.contiguous().to(torch.float32)

        # Dimensions
        B, S, N = x_f32.shape
        grid = B * S

        # Allocate outputs for mean and std
        mean = torch.empty(grid, dtype=torch.float32, device=x_f32.device)
        std = torch.empty(grid, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel
        _row_reduce_mean_std_kernel[grid](
            x_f32, mean, std, N,
            BLOCK_SIZE=8192,
            num_warps=8, num_stages=4
        )

        # Allocate device scalar for std_multiplier and compute via Triton
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=x_f32.device)
        std_multiplier = torch.empty([1], dtype=torch.float32, device=x_f32.device)

        _ndtri_scalar_kernel[0](
            p_dev, std_multiplier
        )

        # Prepare output
        out_f32 = torch.empty_like(x_f32)

        # Launch elementwise sparsification + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_f32, mean, std, out_f32, N,
            std_multiplier[0],
            BLOCK_SIZE=1024,
            num_warps=4, num_stages=3
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
