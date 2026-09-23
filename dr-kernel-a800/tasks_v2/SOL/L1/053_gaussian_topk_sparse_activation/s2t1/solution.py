import torch
import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(prob_ptr, out_ptr, iters: tl.constexpr):
    """
    Triton kernel to compute inverse standard normal CDF for p = prob_ptr[0].
    Uses bisection on the standard normal CDF with an erf approximation.
    Writes result to out_ptr[0] as float32.
    """
    p = tl.load(prob_ptr)  # scalar probability in (0, 1)
    low = -6.0
    high = 6.0
    # Track current mid where CDF(mid) ~ p
    result = tl.zeros((), dtype=tl.float32)
    for _ in range(iters):
        mid = 0.5 * (low + high)
        # CDF of standard normal: 0.5 * (1 + erf(mid / sqrt(2)))
        sqrt2 = 1.4142135623730951
        z = mid / sqrt2
        az = tl.abs(z)
        # Abramowitz & Stegun-style erf approximation:
        # erf(x) ≈ sign(x) * (1 - P(t) * exp(-x^2)), t = 1 / (1 + p*|x|), P(t) is a 5th-degree polynomial
        p_const = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        t = 1.0 / (1.0 + p_const * az)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        e = tl.exp(-(az * az))
        erf_approx = 1.0 - poly * e
        erf_mid = tl.where(z >= 0, erf_approx, -erf_approx)
        cdf = 0.5 * (1.0 + erf_mid)
        result = tl.where(p < cdf, mid, result)
        high = tl.where(p < cdf, mid, high)
        low = tl.where(p >= cdf, mid, low)
    tl.store(out_ptr, result)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per (batch, seq) row. Computes:
      - mean and std over the last dimension N (float32)
      - threshold = mean + std * std_multiplier
      - output = max(0, x - threshold) for each element in the row
    x_ptr points to a contiguous [B, S, N] tensor; each program handles one row starting at base = pid * N.
    """
    pid = tl.program_id(axis=0)
    base = pid * N

    # Accumulators in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: reduction over N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to roundoff
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: elementwise gating
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU: max(0, y)
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward. All computation is done inside Triton kernels.
        - If target_sparsity == 0.0, return inputs unchanged.
        - Else, compute per-row mean/std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold), returning in bfloat16.
        """
        # Early exit for no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Expect 3D input [B, S, N]
        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")
        B, S, N = inputs.shape

        # Ensure contiguous and use float32 for kernel math
        inputs_f32 = inputs.contiguous().to(torch.float32)

        # Output tensor in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=inputs.device)

        # Compute inv-Phi(target_sparsity) using Triton (scalar on device)
        prob = torch.tensor([target_sparsity], dtype=torch.float32, device=inputs.device)
        invphi = torch.empty((), dtype=torch.float32, device=inputs.device)
        compute_invphi_kernel[(1,)](prob, invphi, iters=20, num_warps=1)

        # Launch row


def run(*args):
    return ModelNew()(*args)
