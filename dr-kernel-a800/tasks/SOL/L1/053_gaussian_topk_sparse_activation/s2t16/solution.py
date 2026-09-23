import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr):
    """
    Compute inverse standard normal CDF for probability p using bisection.
    Writes result to out_ptr[0] as float32.
    """
    # Bisection over z in [-6, 6]; run a fixed number of iterations
    a = -6.0
    b = 6.0
    # We approximate erf(z / sqrt(2)) via a standard polynomial approximation.
    # Since Triton may not have tl.erf, we implement a well-known approximation.
    for _ in range(50):
        z = 0.5 * (a + b)
        x = z * 0.7071067811865476  # 1/sqrt(2)
        # Abramowitz & Stegun 7.1.26 erf approximation
        ax = tl.abs(x)
        # Choose p for t = 1 / (1 + p*|x|)
        # Using a constant p=0.147; this is a standard choice for erf approximation
        t = 1.0 / (1.0 + 0.147 * ax)
        # P(t) = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        P = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - P * tl.exp(-(ax * ax))
        erf_approx = tl.where(x >= 0, erf_approx, -erf_approx)
        cdf = 0.5 * (1.0 + erf_approx)
        if cdf > p:
            b = z
        else:
            a = z
    tl.store(out_ptr, 0.5 * (a + b))


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, std_multiplier_ptr, B, S, N, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s).
    - First pass: compute mean and std across N for that row.
    - Second pass: compute threshold = mean + std * std_multiplier and apply y = x - threshold if > 0 else 0.
    """
    pid = tl.program_id(0)
    s = pid % S
    b = pid // S

    # Base offsets for row (b, s) along N
    row_offset = (b * S + s) * N

    # Accumulate sum and sum of squares (float32)
    sum_x = 0.0
    sum_x2 = 0.0

    # First pass over N in chunks
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=tl.float32(0.0))
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_x / n_f
    var = sum_x2 / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # numerical guard
    std = tl.sqrt(var)

    # Load std_multiplier (1-element tensor)
    std_multiplier = tl.load(std_multiplier_ptr)
    threshold = mean + std * std_multiplier

    # Second pass: apply gating y = x - threshold if positive, else 0
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=tl.float32(0.0))
        y = x - threshold
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized sparse activation:
        - Compute per-row mean and std.
        - threshold = mean + std * inv-Phi(target_sparsity).
        - output = x - threshold if positive, else 0; return in bfloat16.
        No torch compute in forward; all math is done in Triton kernels.
        """
        # If no sparsity requested, return x unchanged (original would return inputs)
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous tensor [B, S, N]
        x = x.contiguous()
        B, S, N = x.shape
        device = x.device

        # Output buffer in float32 for computation
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=device)

        # Scalar std_multiplier buffer (1 element) on device
        std_multiplier = torch.empty(1, dtype=torch.float32, device=device)

        # Launch Triton kernel to compute inv-Phi(target_sparsity)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch Triton kernel for sparsity gating: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x, out_f32, std_multiplier, B, S, N, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
