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
    p = tl.load(p_dev_ptr)

    # Constants for A&S approximation (7.1.26)
    # Lower region constants
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

    # Upper region constants
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Compute region flags
    use_low = p < p_low

    # Lower region approximation
    z_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1*z_low + c2)*z_low + c3)*z_low + c4)*z_low + c5)*z_low + c6)
    poly_den_low = (((((d1*z_low + d2)*z_low + d3)*z_low + d4)*z_low + 1.0))
    x_low = poly_low / poly_den_low

    # Central region
    z_mid = p - 0.5
    r_mid = z_mid * z_mid
    poly_mid = (((((a1*r_mid + a2)*r_mid + a3)*r_mid + a4)*r_mid + a5)*r_mid + a6)
    poly_den_mid = (((((b1*r_mid + b2)*r_mid + b3)*r_mid + b4)*r_mid + b5)*r_mid + 1.0)
    x_mid = poly_mid * z_mid / poly_den_mid

    # Upper region
    z_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1*z_high + c2)*z_high + c3)*z_high + c4)*z_high + c5)*z_high + c6)
    poly_den_high = (((((d1*z_high + d2)*z_high + d3)*z_high + d4)*z_high + 1.0))
    x_high = -poly_high / poly_den_high

    # Select result based on region
    result = tl.where(use_low, x_low, tl.where(p <= p_high, x_mid, x_high))

    # Store to out_ptr[0]
    tl.store(out_ptr, result)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(inp - (mean + std * std_multiplier)).
    inp_ptr: *f32, shape [B, S, N]
    out_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    std_multiplier: f32 scalar computed in _ndtri_scalar_kernel
    """
    pid = tl.program_id(0)
    row_start = pid * N
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * std_multiplier

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
        Triton-optimized version of the original run function.
        - Computes mean and std per (batch, seq) row using Triton reduction.
        - Computes inverse normal CDF for target_sparsity via Triton scalar kernel.
        - Applies adaptive threshold + ReLU sparsification using Triton elementwise kernel.
        """
        # Handle no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Work in float32 for numerical stability
        inputs_f32 = inputs.to(torch.float32).contiguous()
        B, S, N = inputs_f32.shape

        # Output buffer
        out_f32 = torch.empty_like(inputs_f32)

        # Allocate mean and std per row
        mean = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel
        # BLOCK_SIZE tuned for typical N up to 16k
        BLOCK_SIZE_RED = 4096
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inputs_f32, mean, std, N, BLOCK_SIZE_RED, num_warps=4, num_stages=2)

        # Compute std_multiplier via Triton scalar kernel
        p_dev = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
        out_dev = torch.empty(1, dtype=torch.float32, device=inputs.device)
        _ndtri_scalar_kernel[(1,)](p_dev, out_dev, num_warps=1, num_stages=1)
        std_multiplier = out_dev[0]  # scalar float

        # Launch sparsification kernel
        # Use larger BLOCK_SIZE for better throughput on big N
        BLOCK_SIZE_ELEM = 8192
        _sparsify_relu_kernel[grid](inputs_f32, out_f32, mean, std, N, std_multiplier, BLOCK_SIZE_ELEM, num_warps=8, num_stages=2)

        # Return in original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
