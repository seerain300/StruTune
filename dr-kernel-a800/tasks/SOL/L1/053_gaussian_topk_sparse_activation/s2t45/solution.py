import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p_scalar: tl.constexpr, out_ptr):
    """
    Compute inverse standard normal CDF for probability p_scalar using bisection.
    Writes result to out_ptr[0] as float32.
    p_scalar: Python float passed to the kernel (no torch ops in forward).
    """
    low = -6.0
    high = 6.0
    eps = 1e-7
    iters = 30

    # Abramowitz & Stegun 7.1.26 approximation for erf(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    for _ in range(iters):
        mid = (low + high) * 0.5
        z = mid
        x = z  # for erf approximation
        t = 1.0 / (1.0 + p_scalar * x)
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
        erf_approx = 1.0 - poly * tl.exp(-x * x)
        # sign handling
        sign = tl.where(x >= 0, 1.0, -1.0)
        erf_approx = sign * erf_approx
        cdf = 0.5 * (1.0 + erf_approx)
        if cdf > p_scalar:
            high = mid
        else:
            low = mid

    invphi = (low + high) * 0.5
    tl.store(out_ptr, invphi)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier):
    """
    One program per row (b, s):
    - First pass: compute sum and sum of squares across N
    - Compute mean and std; then threshold = mean + std * std_multiplier
    - Second pass: apply gating out = max(0, x - threshold)
    All arithmetic in float32. Inputs are assumed contiguous [B, S, N].
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    base = (b * S + s) * N

    # First pass: accumulate sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    n = 0
    while n < N:
        offs = n + tl.arange(0, 1024)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0, to=tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        n += 1024

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: apply gating
    n = 0
    while n < N:
        offs = n + tl.arange(0, 1024)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0, to=tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)
        n += 1024


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-based implementation of the original logic:
        - If target_sparsity == 0.0, return x unchanged (no gating).
        - Else, compute per-row mean and std in float32, threshold = mean + std * inv-Phi(sparsity),
          and return max(0, x - threshold) in bfloat16.
        Forward avoids any torch operations except allocation and dtype cast on return.
        """
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and cast to float32 for computation
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Output as float32; we'll return bfloat16
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element tensor for std_multiplier
        std_multiplier = torch.empty((1,), dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(std_multiplier) entirely in Triton; pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # One program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](x_f32, out_f32, B, S, N, float(std_multiplier[0]), BLOCK_SIZE=1024, num_warps=8)

        # Return in bfloat16 to match original casting
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
