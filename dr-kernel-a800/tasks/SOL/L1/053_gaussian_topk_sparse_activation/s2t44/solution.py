import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF for probability p in (0, 1) and write to out_ptr[0].
    Uses bisection over z in [-6, 6] with Abramowitz & Stegun erf approximation (7.1.26).
    """
    low = -6.0
    high = 6.0
    # Iterate until convergence: (high - low) < 1e-7
    for _ in range(100):
        mid = 0.5 * (low + high)
        # erf approximation: erf(x) ≈ sign(x) * (1 - poly(x) * exp(-x^2))
        ax = tl.abs(mid)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        # Coefficients for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-ax * ax)  # since sign(mid)=+1 for all iterations
        # Standard normal CDF
        cdf = 0.5 * (1.0 + erf_approx * tl.sign(mid))
        # Adjust bounds
        if cdf > p:
            high = mid
        else:
            low = mid
    res = (low + high) * 0.5
    tl.store(out_ptr, res)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B: tl.constexpr, S: tl.constexpr, N: tl.constexpr, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per row (b, s). Two passes over N:
    - First pass: compute mean and std in float32.
    - Second pass: compute threshold and apply gating y = max(0, x - threshold), store as float32.
    x_ptr and out_ptr are assumed to point to float32 data.
    """
    row_id = tl.program_id(axis=0)  # 0 .. B*S-1
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N

    # First pass: sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)  # avoid tiny negative due to round-off
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: gating
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of Gaussian-based top-k sparse activation.
        Computes mean and std per [B, S] row, derives threshold = mean + std * inv-Phi(target_sparsity),
        applies ReLU gating: max(0, x - threshold), returns bfloat16.
        """
        # Handle no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and cast to float32 for robust statistics
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape

        # Allocate 1-element buffer for inv-phi result (float32)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch kernel to compute inv-Phi(target_sparsity)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Prepare output buffer in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Launch kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32,
            B=B, S=S, N=N,
            std_multiplier=std_multiplier[0],
            BLOCK_SIZE=1024,
            num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
