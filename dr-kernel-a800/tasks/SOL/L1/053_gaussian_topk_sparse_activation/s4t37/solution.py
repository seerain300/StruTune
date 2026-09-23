import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr,
                                 B, S, N,
                                 stride_b, stride_s, stride_n,
                                 BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, base pointer to [B, S, N] contiguous tensor
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    B, S, N: int, dimensions
    stride_b, stride_s, stride_n: int, strides for (batch, seq, feature) dims
    """
    pid = tl.program_id(0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S
    row_start = b * stride_b + s * stride_s  # pointer offset for the start of this row

    # Accumulators
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks of BLOCK_SIZE
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx * stride_n, mask=mask, other=0.0)
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
    p_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    p = tl.load(p_ptr)  # scalar

    # A&S constants for the approximation
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Compute regions
    low = p < p_low
    mid = (p >= p_low) & (p <= p_high)
    high = p > p_high

    z = 0.0

    # Lower region approximation
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region approximation
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = (poly * q_mid) / poly2

    # Upper region approximation
    q_hi = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_hi = -(((((c1 * q_hi + c2) * q_hi + c3) * q_hi + c4) * q_hi + c5) * q_hi + c6) / \
           ((((d1 * q_hi + d2) * q_hi + d3) * q_hi + d4) * q_hi + 1.0)

    # Select region result
    z = tl.where(low, z_low, z)
    z = tl.where(mid, z_mid, z)
    z = tl.where(high, z_hi, z)

    tl.store(out_ptr, z)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, out_ptr,
                          mean_ptr, std_ptr,
                          B, S, N,
                          stride_b, stride_s, stride_n,
                          std_multiplier,  # scalar float
                          BLOCK_SIZE: tl.constexpr):
    """
    Elementwise: out = max(0, inp - (mean + std * std_multiplier))
    per (batch, seq) row across N features.
    """
    pid = tl.program_id(0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S
    row_start = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * std_multiplier

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx * stride_n, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(out_ptr + row_start + idx * stride_n, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run:
        - Compute per-(batch,seq) mean and std across feature dim (last).
        - Compute inverse standard normal CDF of target_sparsity in Triton.
        - Apply adaptive threshold: out = relu(x - (mean + std * std_multiplier)).
        """
        # If no sparsity requested, return original
        if target_sparsity == 0.0:
            return inputs

        # Work in float32 for statistics; keep input contiguous
        inputs_f32 = inputs.to(torch.float32).contiguous()
        B, S, N = inputs_f32.shape

        # Allocate outputs and stats
        out_f32 = torch.empty_like(inputs_f32)
        mean = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s) row
        BLOCK_SIZE_RED = 4096
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](
            inputs_f32, mean, std,
            B, S, N,
            inputs_f32.stride(0), inputs_f32.stride(1), inputs_f32.stride(2),
            BLOCK_SIZE_RED,
            num_warps=4, num_stages=2
        )

        # Compute std_multiplier using Triton scalar kernel
        p_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_buf[0] = float(target_sparsity)  # assign scalar from host
        out_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        _ndtri_scalar_kernel[(1,)](p_buf, out_buf, num_warps=1, num_stages=1)
        std_multiplier = float(out_buf[0].item())

        # Launch elementwise sparsification kernel
        BLOCK_SIZE_ELEM = 8192
        _sparsify_relu_kernel[grid](
            inputs_f32, out_f32,
            mean, std,
            B, S, N,
            inputs_f32.stride(0), inputs_f32.stride(1), inputs_f32.stride(2),
            std_multiplier,
            BLOCK_SIZE_ELEM,
            num_warps=8, num_stages=2
        )

        # Cast back to original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
