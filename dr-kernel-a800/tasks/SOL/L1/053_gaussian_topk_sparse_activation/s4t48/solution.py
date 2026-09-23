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

    # Accumulators (float32)
    sum_x = 0.0
    sum_x2 = 0.0

    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        off += BLOCK_SIZE

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _ndtri_scalar_kernel(p_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Loads p from p_ptr[0], stores result to out_ptr[0].
    Expresses piecewise logic using tl.where; no Python conditionals on Triton values.
    """
    p = tl.load(p_ptr)  # scalar float32

    # A&S constants for lower and upper branches
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

    # Central region constants (A&S 7.1.26)
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

    # Branching thresholds
    p_low = 0.02425
    p_mid_start = 0.5
    p_mid_end = 0.5 + p_low

    # Lower branch: p < 0.5
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Upper branch: p > 0.5
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Central branch: 0.5 <= p <= 0.5 + p_low
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    poly_denom = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / poly_denom

    # Select branches using masks (no Python conditionals on Triton values)
    mask_low = p < 0.5
    mask_high = p > 0.5
    z = tl.where(mask_low, z_low, z_mid)
    z = tl.where(mask_high, z_up, z)

    tl.store(out_ptr, z)


@triton.jit
def _sparsify_row_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification for a row:
    out[row, :, :] = relu(inp[row, :, :] - (mean[row] + std[row] * std_multiplier))
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load per-row stats
    mean = tl.load(mean_ptr + pid)  # float32
    std = tl.load(std_ptr + pid)    # float32

    # Compute threshold (std_multiplier is a scalar tensor, load as scalar)
    threshold = mean + std * std_multiplier

    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)
        off += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation.
        For each (batch, seq) row, compute:
          mean = avg(row), std = sqrt(E[row^2] - mean^2) (population std, unbiased=False)
        threshold = mean + std * norm.icdf(target_sparsity)
        output = relu(inputs - threshold)
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs.clone()

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)
        B, S, N = inp_f32.shape
        device = inp_f32.device

        # Allocate per-row stats
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch reduction kernel
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inp_f32, mean, std, N, BLOCK_SIZE=2048, num_warps=8, num_stages=4)

        # Compute std multiplier via Triton scalar kernel
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
        z_tensor = torch.empty([1], dtype=torch.float32, device=device)
        _ndtri_scalar_kernel[(1,)](p_tensor, z_tensor)
        std_multiplier = z_tensor[0]  # scalar tensor on device

        # Allocate output
        out_f32 = torch.empty_like(inp_f32)

        # Launch elementwise sparsification kernel
        _sparsify_row_kernel[grid](inp_f32, mean, std, out_f32, N, std_multiplier, BLOCK_SIZE=4096, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
