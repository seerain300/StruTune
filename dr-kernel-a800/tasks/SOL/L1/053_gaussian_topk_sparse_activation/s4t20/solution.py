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
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p (scalar)
    p = tl.load(p_dev_ptr)

    # Constants for A&S approximation (7.1.26)
    p_low = 0.02425
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
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    z = 0.0
    if p < p_low:
        # Lower region: z = sqrt(2) * sqrt(-log(p))
        z = 1.4142135623730951 * tl.sqrt(-tl.log(p))
        z = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) / \
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0)
    else:
        # Upper region: p > 1 - p_low
        if p > 1.0 - p_low:
            q = 1.4142135623730951 * tl.sqrt(-tl.log(1.0 - p))
            z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        else:
            # Central region
            q = p - 0.5
            r = q * q
            z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Store the result (negative z corresponds to lower tail)
    tl.store(out_ptr, z)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(inp - (mean + std * std_multiplier))
    inp_ptr: *f32, shape [B*S, N] flattened
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B*S, N]
    N: int, length of last dim
    std_multiplier: f32 scalar (result of ndtri)
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load scalar mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)

    cutoff = mean + std * std_multiplier

    # Iterate over row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run() from the original Model.
        - All device tensor math is done in Triton kernels (no torch ops on device tensors).
        - Uses float32 for compute; returns in bfloat16 (matching original behavior).
        """
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and cast to float32 for computation
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Shapes
        B, S, N = x_f32.shape
        RS = B * S

        # Allocate output
        out_f32 = torch.empty_like(x_f32)

        # Allocate mean and std buffers
        mean = torch.empty(RS, dtype=torch.float32, device=x_f32.device)
        std = torch.empty(RS, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (B,S) row
        grid = (RS,)
        _row_reduce_mean_std_kernel[grid](
            x_f32, mean, std, N,
            BLOCK_SIZE=4096,
            num_warps=8, num_stages=4
        )

        # Compute scalar inverse-normal for target_sparsity using Triton
        p_dev = torch.tensor(target_sparsity, dtype=torch.float32, device=x_f32.device)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier)  # single program

        # Launch elementwise sparsification + ReLU kernel
        _sparsify_relu_kernel[grid](
            x_f32, mean, std, out_f32, N,
            std_multiplier[0],
            BLOCK_SIZE=1024,
            num_warps=4, num_stages=3
        )

        # Cast back to original dtype (original code returns bfloat16)
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
