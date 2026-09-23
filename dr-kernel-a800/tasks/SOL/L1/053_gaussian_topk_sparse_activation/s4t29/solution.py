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
def _ndtri_scalar_kernel(p_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Loads p from device tensor p_ptr (shape [1]), writes result to out_ptr[0].
    p_ptr: *f32, shape [1]
    out_ptr: *f32, shape [1]
    """
    # Load scalar p
    p = tl.load(p_ptr)

    # Constants for A&S 7.1.26 approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p_low = 0.02425

    # Determine regions
    p_gt_half = p > 0.5
    p_lt_half = p < 0.5
    # For upper region, use symmetry: ndtri(1-p) = -ndtri(p)
    use_upper = p_gt_half
    # Compute q for lower region
    q = tl.sqrt(-2.0 * tl.log(p))  # p in (0, 0.5) -> log(p) is fine
    # Polynomial in q
    poly = a1 * q + a2 * q * q + a3 * q * q * q + a4 * q * q * q * q + a5 * q * q * q * q * q
    z_lower = poly * tl.exp(-0.5 * q * q)  # note: using exp(-q^2/2) in the polynomial formula
    # For upper region, z_upper = -z_lower (symmetry)
    z_upper = -z_lower
    # Apply piecewise
    z = tl.where(use_upper, z_upper, z_lower)
    # Write result
    tl.store(out_ptr, z)


@triton.jit
def _sparsify_kernel(x_ptr, mean_ptr, std_ptr, out_ptr, N, multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification per row: out = max(x - (mean + std * multiplier), 0).
    x_ptr: *f32, shape [B, S, N], contiguous input
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    multiplier: f32 scalar computed from target_sparsity
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * multiplier

    # Iterate over the row and apply ReLU(x - cutoff)
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the adaptive sparsity activation.
        Computes per-row mean and std, multiplies std by inverse normal CDF of target_sparsity via Triton,
        and applies out = max(input - (mean + std * multiplier), 0).
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
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

        # Launch reduction kernel (tuned)
        BLOCK_SIZE_R = 2048  # reduce loop iterations for N up to 16k
        grid = (total,)
        _row_reduce_mean_std_kernel[grid](x32, mean, std, N, BLOCK_SIZE_R, num_warps=8, num_stages=4)

        # Compute std_multiplier on device via Triton (A&S 7.1.26)
        # Create device tensor for p (1-element)
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
        out_dev = torch.empty([1], dtype=torch.float32, device=device)
        _ndtri_scalar_kernel[(1,)](p_dev, out_dev)  # single program
        std_multiplier = float(out_dev.item())  # read scalar to host for broadcast; cheap

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
