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
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p
    p = tl.load(p_dev_ptr)

    # Constants (Abramowitz & Stegun 7.1.26)
    # Lower region
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Central region
    p_low = 0.02425
    p_high = 1.0 - p_low

    q = p - 0.5
    r = q * q

    central = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    ndtri_p = central / denom

    # Store result
    tl.store(out_ptr, ndtri_p)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise: out = relu(inp - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B*S, N], contiguous per row
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B*S, N]
    N: int, last dim length
    std_multiplier_ptr: *f32, shape [1] (device scalar)
    BLOCK_SIZE: constexpr chunk size along N
    """
    row_id = tl.program_id(0)  # one program per (batch, seq) row
    col_block = tl.program_id(1)  # block along N
    row_start = row_id * N
    col_start = col_block * BLOCK_SIZE

    idx = col_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # Load x for this row block
    x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    multiplier = tl.load(std_multiplier_ptr)  # scalar

    threshold = mean + std * multiplier
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return as is
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32
        x_in = x.contiguous()
        x_f32 = x_in.to(torch.float32)

        B, S, N = x_f32.shape
        BS = B * S

        # Allocate mean and std buffers
        mean = torch.empty(BS, dtype=torch.float32, device=x_f32.device)
        std = torch.empty(BS, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per row
        grid_reduce = (BS,)
        _row_reduce_mean_std_kernel[grid_reduce](
            x_f32, mean, std, N,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=4
        )

        # Allocate device scalar p and output for std_multiplier
        p_dev = torch.empty(1, dtype=torch.float32, device=x_f32.device)
        p_dev[0] = float(target_sparsity)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Compute inverse normal CDF in Triton
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier)

        # Prepare output buffer
        out_f32 = torch.empty_like(x_f32)

        # Launch elementwise sparsification + ReLU: grid over rows and blocks along N
        BLOCK = 1024
        grid = (BS, triton.cdiv(N, BLOCK))
        _sparsify_relu_kernel[grid](
            x_f32, mean, std, out_f32, N, std_multiplier,
            BLOCK_SIZE=BLOCK,
            num_warps=4, num_stages=3
        )

        # Cast back to original dtype
        return out_f32.to(x.dtype)


def run(*args):
    return ModelNew()(*args)
