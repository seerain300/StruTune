import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_sum_and_sumsq_kernel(inp_ptr, sum_ptr, sumsq_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute sum and sum of squares across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks
    num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        off = chunk * BLOCK_SIZE
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        # Reduce within the vector to scalars and accumulate
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    tl.store(sum_ptr + pid, sum_x)
    tl.store(sumsq_ptr + pid, sum_x2)


@triton.jit
def _compute_mean_std_kernel(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, N):
    """
    For each (batch, seq) row, compute mean and std across the last dimension using precomputed sum and sumsq.
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    sum_x = tl.load(sum_ptr + pid)
    sum_x2 = tl.load(sumsq_ptr + pid)
    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S 7.1.26 approximation.
    Loads p from p_dev_ptr (shape [1]) and stores result to out_ptr[0].
    """
    p = tl.load(p_dev_ptr)
    # A&S constants
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

    # Lower region approximation
    # q = sqrt(-2 * log(p)) with log for p in (0, p_low)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    phi_low = poly_low / den_low

    # Central region approximation
    # q = p - 0.5
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    phi_mid = poly_mid / den_mid

    # Upper region approximation (for p > 1 - p_low)
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    den_up = (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
    phi_up = -poly_up / den_up

    # Select region and combine
    # For p < 0.5: phi_low, else for p > 0.5: phi_up, else: phi_mid
    # Using piecewise selection in Triton
    mask_low = p < p_low
    mask_up = p > 0.5
    # Initialize output
    phi = tl.zeros_like(p)

    # Apply masks; Triton supports masked assignment
    phi = tl.where(mask_low, phi_low, phi)
    phi = tl.where(mask_up, phi_up, phi)
    # remaining (p in [p_low, 0.5]): phi_mid
    # Note: p_mid = (p >= p_low) & (p <= 0.5)
    p_mid = (p >= p_low) & (p <= 0.5)
    phi = tl.where(p_mid, phi_mid, phi)

    tl.store(out_ptr, phi)


@triton.jit
def _sparsify_row_kernel(inp_ptr, out_ptr, mean_ptr, std_ptr, multiplier_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise kernel: out = relu(inp - (mean + std * multiplier))
    inp_ptr: *f32, shape [B, S, N], contiguous
    out_ptr: *f32, same shape
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    multiplier_ptr: *f32, shape [1]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N
    # Load per-row stats
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    scale = tl.load(multiplier_ptr)  # scalar
    cutoff = mean + std * scale

    # Process the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
          - If target_sparsity == 0.0: return inputs.
          - Else: compute per-row sum and sum of squares (Triton), compute mean and std (simple torch ops),
                  compute inverse-normal multiplier (Triton), then apply adaptive threshold ReLU (Triton).
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous float32 for computation
        inputs = inputs.contiguous()
        inputs_f32 = inputs.to(torch.float32)

        B, S, N = inputs_f32.shape

        # Allocate per-row sums and sumsq (reduce across last dim)
        sum_x = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sum_x2 = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel to compute sum and sumsq
        BLOCK_SIZE = 2048  # tuned for typical sizes up to 16384
        grid = (B * S,)
        _row_reduce_sum_and_sumsq_kernel[grid](inputs_f32, sum_x, sum_x2, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute per-row mean and std with simple torch ops (on small vectors)
        mean = sum_x / float(N)
        std = torch.sqrt(sum_x2 / float(N) - mean * mean)

        # Compute inv-std-normal multiplier for target_sparsity using Triton (scalar)
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        std_multiplier = torch.empty([1], dtype=torch.float32, device=inputs.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier)

        # Allocate output
        out = torch.empty_like(inputs_f32)

        # Launch sparsification kernel
        _sparsify_row_kernel[grid](inputs_f32, out, mean, std, std_multiplier, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Cast back to original dtype
        return out.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
