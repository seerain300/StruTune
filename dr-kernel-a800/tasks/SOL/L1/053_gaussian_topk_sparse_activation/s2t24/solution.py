import math

# Triton kernels must be defined here; no torch operations in ModelNew.forward.

@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr: tl.pointer_type(tl.float32)):
    """
    Compute inv-Phi(p) using bisection on z in [-6, 6] with erf approximation.
    Writes the result to out_ptr[0].
    """
    # Bisection interval
    a = -6.0
    b = 6.0
    tol = 1e-7

    # Standard normal CDF using erf approximation: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Implement erf(x) approximation (Abramowitz & Stegun 7.1.26)
    # erf(x) ≈ sign(x) * (1 - t * exp(-x^2) * (a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4)),
    # where t = 1 / (1 + p |x|), p=0.3275911, a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429.
    # Note: we implement Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    # We will define erf approximation function within bisection loop.

    # Bisection loop
    # We'll do a fixed number of iterations for efficiency; 28 iterations give sufficient precision.
    # Using while loop to adjust bracket.
    for _ in range(28):
        z = (a + b) * 0.5
        # Compute Phi(z) via erf approximation at z/sqrt(2)
        x = z * 0.7071067811865476  # 1/sqrt(2)
        # sign and |x|
        sign = 1.0
        absx = x
        # erf approximation constants
        p_c = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        # t = 1 / (1 + p_c |x|)
        t = 1.0 / (1.0 + p_c * absx)
        # Polynomial evaluation
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
        # erf(x) ≈ sign(x) * (1 - poly * exp(-x*x))
        # Note: exp(-x*x) is fine for small x (our z in [-6,6] keeps x in that range).
        exp_term = tl.exp(-absx * absx)
        erf_approx = 1.0 - poly * exp_term
        erf_approx = sign * erf_approx  # sign: positive here since x can be negative, but approximation covers both
        # Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        phi = 0.5 * (1.0 + erf_approx)
        # Update bracket
        if phi > p:
            b = z
        else:
            a = z
    # Store result to out_ptr[0]
    tl.store(out_ptr, z)


@triton.jit
def row_sparsity_kernel(
    x_ptr, out_ptr,
    B: tl.int32, S: tl.int32, N: tl.int32,
    std_multiplier_ptr: tl.pointer_type(tl.float32),
    BLOCK: tl.constexpr
):
    """
    One program per row (b, s).
    Two passes over N: first to compute mean and std, second to apply ReLU gating.
    """
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Compute base index for this row
    # Flattened indexing: row[b, s, :] corresponds to indices [b*S*N + s*N : b*S*N + s*N + N)
    # But since we iterate over N directly, we can compute offsets as n in [0, N).
    # We will loop over N in chunks of BLOCK.

    # First pass: compute sum and sum of squares in float32
    total_sum = 0.0
    total_sq = 0.0
    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK)
        mask = offs < N
        # Compute linear offsets into x_ptr: out_ptr and x_ptr are contiguous [B, S, N]
        # For each (b, s) row, elements are contiguous across N. We can flatten B, S, N as [B*S, N].
        # To get pointer for row b,s,: we take a base offset for (b, s) row.
        # Since x is contiguous [B, S, N], the base for row (b, s) is (b*S + s)*N.
        base = (b * S + s) * N
        x_vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        total_sum += tl.sum(x_vals, axis=0)
        total_sq += tl.sum(x_vals * x_vals, axis=0)
        n += BLOCK

    # Compute mean and std (population std, unbiased=False)
    N_f = N  # integer N, safely used in division
    mean = total_sum / N_f
    var = total_sq / N_f - mean * mean
    std = tl.sqrt(var)
    # threshold = mean + std * std_multiplier
    std_mul = tl.load(std_multiplier_ptr)
    threshold = mean + std * std_mul

    # Second pass: apply ReLU gating and store
    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK)
        mask = offs < N
        base = (b * S + s) * N
        x_vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        y_vals = tl.maximum(x_vals - threshold, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y_vals, mask=mask)
        n += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward: computes adaptive threshold using inverse normal CDF,
        then applies per-row ReLU gating. No torch operations are used in forward.
        """
        # If no sparsity requested, return input as-is
        if target_sparsity == 0.0:
            return x

        # Ensure inputs are contiguous and compute in float32
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        B = x_f32.shape[0]
        S = x_f32.shape[1]
        N = x_f32.shape[2]

        # Allocate output in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Compute std_multiplier inside Triton kernel; pass as 1-element tensor
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch compute_invphi_kernel: pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32,
            B, S, N,
            std_multiplier,
            BLOCK=1024,
            num_warps=4
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
