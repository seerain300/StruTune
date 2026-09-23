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
    p_dev_ptr: *f32, shape [1] (device tensor holding scalar target_sparsity)
    out_ptr: *f32, shape [1] (device tensor to store scalar result)
    """
    # Load p
    p = tl.load(p_dev_ptr)

    # Constants for A&S 7.1.26 approximation
    p_low = 0.02425
    # Central region constants
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

    # Lower region
    mask_low = p < p_low
    # For lower region: q = sqrt(-2 * log(p)); result = polynomial(q) / polynomial(q+1)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1*q_low + c2)*q_low + c3)*q_low + c4)*q_low + c5)*q_low + c6) / \
                 ((((d1*q_low + d2)*q_low + d3)*q_low + d4)*q_low + 1.0)

    # Central region (p >= p_low and p <= 1 - p_low)
    p_mid = p - 0.5
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    r = p_mid * p_mid
    result_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*p_mid / \
                 (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)

    # Upper region
    mask_high = p > (1.0 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_high = -(((((c1*q_high + c2)*q_high + c3)*q_high + c4)*q_high + c5)*q_high + c6) / \
                  ((((d1*q_high + d2)*q_high + d3)*q_high + d4)*q_high + 1.0)

    # Combine regions
    result = tl.where(mask_low, result_low, 0.0)
    result = tl.where(mask_mid, result_mid, result)
    result = tl.where(mask_high, result_high, result)

    # Store to output
    tl.store(out_ptr, result)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, threshold_scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(input - (mean + std * threshold_scale))
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    threshold_scale: f32 scalar computed in Triton ndtri kernel
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Load mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * threshold_scale

    # Iterate over the row in chunks and apply ReLU(input - cutoff)
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the adaptive sparsity activation.
        Computes mean and std per (batch, seq) row, applies ReLU(input - (mean + std * ndtri(sparsity))).
        """
        # Handle no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32 for stability
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)
        B, S, N = inp_f32.shape
        total_rows = B * S

        # Allocate outputs and stats
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out_f32 = torch.empty_like(inp_f32)

        # Compute inverse normal CDF for target sparsity in Triton (scalar)
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inp_f32.device)
        out_scale = torch.empty([1], dtype=torch.float32, device=inp_f32.device)
        _ndtri_scalar_kernel[(1,)](p_tensor, out_scale, num_warps=1, num_stages=1)
        threshold_scale = out_scale[0]

        # Launch reduction kernel
        BLOCK_SIZE_R = 1024  # good default; adjust if needed for very large N
        _row_reduce_mean_std_kernel[(total_rows,)](inp_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE_R, num_warps=4, num_stages=2)

        # Launch elementwise sparsification kernel
        BLOCK_SIZE_E = 1024
        _sparsify_relu_kernel[(total_rows,)](inp_f32, mean, std, out_f32, N, threshold_scale, BLOCK_SIZE=BLOCK_SIZE_E, num_warps=4, num_stages=2)

        # Cast back to original dtype
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
