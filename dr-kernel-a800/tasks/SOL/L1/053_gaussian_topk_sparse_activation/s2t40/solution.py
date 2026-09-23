import math

# Triton kernels
@triton.jit
def compute_invphi_kernel(p: float, out_ptr: tl.pointer_type(tl.float32), iters: tl.constexpr):
    """
    Compute inverse standard normal CDF for p in (0, 1) and write to out_ptr[0].
    Uses bisection over z in [-6, 6] and an erf approximation (Abramowitz & Stegun 7.1.26).
    """
    # Choose sign based on p
    neg = True
    if p > 0.5:
        neg = False
        p = 1.0 - p

    lo = -6.0
    hi = 6.0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ax = tl.abs(mid)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        # Abramowitz & Stegun 7.1.26 polynomial approximation for erf(mid)
        poly = (((((0.1735111) * t + 0.6837933) * t + 1.453152) * t + 1.5174358) * t + 1.0121072)
        erf_mid = 1.0 - poly * tl.exp(-ax * ax)
        if mid < 0.0:
            erf_mid = -erf_mid
        cdf = 0.5 * (1.0 + erf_mid)
        if p < cdf:
            hi = mid
        else:
            lo = mid

    invphi = 0.5 * (lo + hi)
    tl.store(out_ptr, invphi)

@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
                         std_multiplier_ptr: tl.pointer_type(tl.float32),
                         BLOCK_SIZE: tl.constexpr):
    """
    One program per (b, s) row. Two-pass:
      - First pass: compute mean and std across N (float32)
      - Second pass: compute threshold and apply ReLU gating
    """
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S
    # Row base offset in flattened [B, S, N] layout: index = ((b * S) + s) * N
    row_base = ((b * S) + s) * N

    # First pass: sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    n_elems = 0
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        n_elems += tl.sum(mask.to(tl.int32), axis=0)

    mean = sum_val / tl.maximum(n_elems, 1)
    ex2 = sum_sq / tl.maximum(n_elems, 1)
    var = ex2 - mean * mean
    var = tl.maximum(var, 0.0)  # clamp for numerical stability
    std = tl.sqrt(var)

    # Load inv-Phi scalar
    std_multiplier = tl.load(std_multiplier_ptr)

    # Second pass: apply gating and store
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        threshold = mean + std * std_multiplier
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + row_base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward: per-row threshold gating with adaptive sparsity based on input statistics.
        Returns bfloat16 tensor.
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous and float32 for computation
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Output buffer (float32 for compute)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Prepare scalar inv-Phi buffer (1 element)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(target_sparsity) in Triton
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier, iters=50)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32,
            B=B, S=S, N=N,
            std_multiplier_ptr=std_multiplier,
            BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 as per original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
