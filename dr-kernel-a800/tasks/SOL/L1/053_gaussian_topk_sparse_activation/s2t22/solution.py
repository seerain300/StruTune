import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr: tl.pointer_type(tl.float32)):
    # Compute inverse standard normal CDF (quantile) for probability p using bisection.
    # Abramowitz & Stegun 7.1.26: erf(x) approximation used to compute CDF.
    # We solve for z such that Phi(z) = p.
    # Bounds: z in [-6, 6].
    # Tolerance: use 1e-7. After ~30 iterations, precision is sufficient for typical needs.

    # Lower bound: compute erf(-z_low) ~ 2*sqrt(2/pi) * (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5), t = 1/(1+p)
    # We need Phi(-6) ~ 1e-7; set z_low = -6.0
    z_low = -6.0
    z_high = 6.0
    tol = 1e-7

    # Newton update per iteration:
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # dPhi/dz = 1 / sqrt(2*pi) * exp(-z^2 / 2)
    # Target = p, f(z) = Phi(z) - p, f'(z) = dPhi/dz
    # z_next = z - f / f'

    # We'll use bisection (safe and fast enough for 1e-7 tolerance) to avoid conditional on sign(f).
    # But since we need to pass p, we can't branch without reading; use a simple loop.

    # Initialize z
    z = 0.0
    # We need to compute for this z at each iteration; we'll recompute erf(z) each iteration.
    # Triton has tl.exp and arithmetic; we'll implement erf approximation inline.

    # Bisection iterations
    for _ in range(28):
        # erf approximation at z/sqrt(2)
        # t = 1/(1 + |z|)
        t = 1.0 / (1.0 + tl.abs(z))
        # poly = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_z = 1.0 - poly * tl.exp(-(z * z))
        # Phi(z)
        phi = 0.5 * (1.0 + erf_z)
        # If phi < p, increase z; else decrease z
        if phi < p:
            z_low = z
        else:
            z_high = z
        # Midpoint
        z = 0.5 * (z_low + z_high)

    # Write result to out_ptr[0]
    tl.store(out_ptr, z)


@triton.jit
def row_sparsity_kernel(
    x_ptr,              # *float32, input flattened rows (B*S*stride_n) accessed by row base + n
    out_ptr,            # *float32, output buffer
    B: tl.int32,        # batch size
    S: tl.int32,        # seq_len
    N: tl.int32,        # intermediate_size
    std_multiplier_ptr, # *float32, 1-element tensor containing inv-Phi(p)
    BLOCK: tl.constexpr
):
    # One program per row: pid in [0, B*S)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base offset for this row (assuming contiguous last dim): row_base = (b*S + s) * N
    row_base = (b * S + s) * N

    # Pass 1: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0, to=tl.float32)
        # Sum and sum of squares for valid elements
        # Since masked loads return 0.0, we need to zero out contributions from invalid lanes.
        # Compute in float32 explicitly.
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        x2 = x * x
        sum_sq += tl.sum(x2, axis=0)
        n += BLOCK

    # Compute mean and std (population std, unbiased=False)
    mean = sum_val / N
    # var = E[x^2] - (E[x])^2
    var = sum_sq / N - mean * mean
    # Guard against tiny negative due to round-off
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (1-element tensor)
    std_mul = tl.load(std_multiplier_ptr)

    # threshold = mean + std * std_mul
    threshold = mean + std * std_mul

    # Pass 2: apply ReLU gating and store
    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0, to=tl.float32)
        x = x.to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_base + offs, y, mask=mask)
        n += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return input unchanged
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32
        x_f32 = x.contiguous().to(torch.float32)

        B, S, N = x_f32.shape

        # Allocate output buffer (float32)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element device buffer for std_multiplier (inv-Phi)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch compute_invphi_kernel: pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK=1024, num_warps=4
        )

        # Return in bfloat16 to match example behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
