import triton
import triton.language as tl


# Kernel 1: compute inverse standard normal CDF for probability p (Abramowitz & Stegun 7.1.26)
# We use bisection over z in [-6, 6] for robustness. Accepts p as a Python float; writes to a 1-element tensor.
@triton.jit
def compute_invphi_kernel(p: tl.float32, out: tl.pointer_type(tl.float32)):
    # 1-element output buffer
    eps = 1e-7
    # Bisection over z in [-6, 6]
    lo = -6.0
    hi = 6.0
    # Number of iterations to reach precision eps
    iters = 0
    while iters < 20:
        mid = 0.5 * (lo + hi)
        # erf approximation (Abramowitz & Stegun 7.1.26)
        # erf(x) ~ sign(x) * (1 - t * exp(-x^2) * poly(t)), t = 1 / (1 + p*|x|)
        x = mid
        ax = tl.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        # Polynomial coefficients
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_mid = 1.0 - poly * tl.exp(-ax * ax)
        erf_mid = tl.where(x >= 0, erf_mid, -erf_mid)
        cdf = 0.5 * (1.0 + erf_mid)
        # Adjust bounds
        if cdf > p:
            hi = mid
        else:
            lo = mid
        iters += 1
    mid = 0.5 * (lo + hi)
    # Write result
    tl.store(out, mid)


# Kernel 2: per-row sparsity computation. One program per (b, s) row. Two passes over N:
#  - first pass: compute mean and std
#  - second pass: compute out = max(0, x - threshold), where threshold = mean + std * std_multiplier
@triton.jit
def row_sparsity_kernel(
    x_ptr,              # *f32, pointer to input
    out_ptr,            # *f32, pointer to output
    B: tl.constexpr,    # int
    S: tl.constexpr,    # int
    N: tl.constexpr,    # int
    std_multiplier_ptr, # *f32, pointer to 1-element tensor holding inv-Phi(target_sparsity)
    BLOCK: tl.constexpr
):
    row_id = tl.program_id(0)  # 0 .. B*S-1
    b = row_id // S
    s = row_id % S
    base = (b * S + s) * N  # index into x/out at row (b, s)

    # First pass: compute sum and sum of squares across N in chunks
    sum_val = 0.0
    sum_sq = 0.0
    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        # Load chunk, cast to f32
        x_chunk = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_chunk = tl.cast(x_chunk, tl.float32)
        sum_val += tl.sum(x_chunk, axis=0)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)
        start += BLOCK

    mean = sum_val / N
    # population std: unbiased=False
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # Load scalar inv-Phi(target_sparsity)
    std_multiplier = tl.load(std_multiplier_ptr)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: compute gated output
    start = 0
    while start < N:
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x_chunk = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_chunk = tl.cast(x_chunk, tl.float32)
        diff = x_chunk - threshold
        gated = tl.maximum(diff, 0.0)
        tl.store(out_ptr + base + offs, gated, mask=mask)
        start += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32
        x = x.contiguous()
        # We will compute in float32 for numerical stability
        x_f32 = x.to(torch.float32)

        B = x_f32.shape[0]
        S = x_f32.shape[1]
        N = x_f32.shape[2]

        # Allocate output tensor (float32 for computation)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element device tensor to hold inv-Phi(target_sparsity)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch compute_invphi_kernel with Python float p
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](x_f32, out_f32, B, S, N, std_multiplier, BLOCK=1024, num_warps=8)

        # Return in bfloat16 to match the original example
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
