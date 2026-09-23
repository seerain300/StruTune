import torch
import triton
import triton.language as tl


# Kernel 1: Per-row reduction to compute mean and std across the last dimension N.
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

    sum_x = 0.0
    sum_x2 = 0.0

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

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


# Kernel 2: Compute inverse standard normal CDF for scalar p via A&S 7.1.26.
@triton.jit
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute z = Phi^{-1}(p) using A&S approximation (7.1.26).
    p_dev_ptr: *f32, shape [1] (device tensor holding scalar p)
    out_ptr: *f32, shape [1] (device tensor to hold scalar result z)
    """
    p = tl.load(p_dev_ptr)

    # Lower region constants
    a1 = -3.9696830e+01
    a2 = 2.2094609e+02
    a3 = -2.7592851e+02
    a4 = 1.3835775e+02
    a5 = -3.0664798e+01
    a6 = 2.5066283e+00

    b1 = -5.4476099e+01
    b2 = 1.6158584e+02
    b3 = -1.5569898e+02
    b4 = 6.6801312e+01
    b5 = -1.3280682e+01

    # Upper region constants
    c1 = -7.7848940e-03
    c2 = -3.2239646e-01
    c3 = -2.4007583e+00
    c4 = -2.5497325e+00
    c5 = 4.3746641e+00
    c6 = 2.9381640e+00

    d1 = 7.7846957e-03
    d2 = 3.2246713e-01
    d3 = 2.4451341e+00
    d4 = 3.7544087e+00

    # Thresholds
    p_low = 0.02425
    p_high = 1.0 - p_low

    use_low = p < 0.5  # if p < 0.5, use lower region; else upper region

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    poly_low = poly_low / (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = -poly_low

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    poly_up = poly_up / (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
    z_up = poly_up

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_mid_num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_mid_den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = poly_mid_num * q_mid / poly_mid_den

    # Select final z
    z = tl.where(use_low, z_low, tl.where(p > 0.5, z_up, z_mid))

    tl.store(out_ptr, z)


# Kernel 3: Elementwise sparsification: out = relu(x - (mean + std * z)).
@triton.jit
def _sparsify_kernel(inp_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification: out = relu(x - (mean + std * z)), where z is scalar.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    z_ptr: *f32, shape [1] (device scalar: inv-std-normal)
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar inv-std-normal
    cutoff = mean + std * z

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        val = x - cutoff
        val = tl.maximum(val, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
            target_sparsity: Float in [0, 1], target sparsity level; 0.0 means no sparsity.

        Returns:
            Sparsified tensor of same shape as input (compute in float32, cast back).
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)

        B, S, N = inp_f32.shape
        total_rows = B * S

        # Allocate stats
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out = torch.empty_like(inp_f32)

        # Launch reduction kernel: one program per row
        BLOCK_SIZE_RED = 2048
        _row_reduce_mean_std_kernel[(total_rows,)](inp_f32, mean, std, N, BLOCK_SIZE_RED, num_warps=8, num_stages=4)

        # Prepare device scalar p and compute inv-std-normal via Triton
        p


def run(*args):
    return ModelNew()(*args)
