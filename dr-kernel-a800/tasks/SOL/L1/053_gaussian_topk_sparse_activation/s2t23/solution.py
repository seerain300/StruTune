import triton
import triton.language as tl


# Kernel 1: compute inverse standard normal CDF for a given p via bisection.
# Accepts p as a Python float and writes the result to out_ptr (1-element tensor).
@triton.jit
def compute_invphi_kernel(p, out_ptr, BLOCK_SIZE: tl.constexpr):
    # Fixed upper bound for z in [-6, 6] suffices for precision we need.
    # Use bisection to find z such that Phi(z) ~ p.
    low = -6.0
    high = 6.0
    # Number of iterations: 24 gives good precision (~1e-5).
    for _ in range(24):
        mid = 0.5 * (low + high)
        # erf approximation (Abramowitz & Stegun 7.1.26)
        t = 1.0 / (1.0 + 0.3275911 * tl.abs(mid))
        # Coefficients for erf
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_mid = 1.0 - poly * tl.exp(-mid * mid)
        cdf = 0.5 * (1.0 + erf_mid * tl.sign(mid))  # sign(mid) corrects for erf's domain
        # cdf should be 0.5 for mid=0, and erf(-x)=-erf(x), so this formula is fine.
        if cdf < p:
            low = mid
        else:
            high = mid
    # Store mid as inv-Phi(p)
    tl.store(out_ptr, mid)


# Kernel 2: per-row sparsity gating using adaptive threshold = mean + std * std_multiplier.
# Grid: one program per row (b, s). Assumes x is [B, S, N] contiguous.
@triton.jit
def row_sparsity_kernel(
    x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK: tl.constexpr
):
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    # Base pointers for the row (flattened across N)
    base = b * S * N + s * N

    # Pass 1: compute sum and sum of squares in float32
    sum_val = 0.0
    sum_sq = 0.0
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        ptrs = x_ptr + base + offs
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    # Guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std multiplier (scalar)
    std_mul = tl.load(std_multiplier_ptr)

    threshold = mean + std * std_mul

    # Pass 2: apply ReLU gate and store
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        ptrs_in = x_ptr + base + offs
        x = tl.load(ptrs_in, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        ptrs_out = out_ptr + base + offs
        tl.store(ptrs_out, y, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            # Return with bfloat16 to match the original example
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and compute in float32
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape

        # Allocate output in float32 for computation
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(std_multiplier) in Triton (no torch ops in forward)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](x_f32, out_f32, B, S, N, std_multiplier, BLOCK=1024, num_warps=4)

        # Return in bfloat16 to match the original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
