import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p_scalar: tl.constexpr, out_ptr: tl.pointer_type(tl.float32)):
    """
    Compute inverse standard normal CDF for probability p_scalar using bisection
    and write the result to out_ptr[0] (1-element float32 tensor).
    """
    # Constants for Abramowitz & Stegun erf approximation
    # We implement inv-Phi via bisection on z in [-6, 6].
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    # Bisection bounds
    low = -6.0
    high = 6.0
    # Tolerance for convergence
    tol = 1e-7

    # Use a fixed number of iterations; bisection converges rapidly
    for _ in range(20):
        z = 0.5 * (low + high)
        # erf(z) approximation via Abramowitz & Stegun 7.1.26
        sign = 1.0 if z >= 0.0 else -1.0
        x = sign * (1.0 - z)
        t = 1.0 / (1.0 + 0.3275911 * x)
        # Horner's method for poly
        poly = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
        erf_approx = sign * (1.0 - poly * tl.exp(-x * x))

        # Standard normal CDF
        phi = 0.5 * (1.0 + erf_approx * 0.7071067811865476)  # 1/sqrt(2)

        if p_scalar > phi:
            low = z
        else:
            high = z

    # Assign the final z to out_ptr[0]
    z = 0.5 * (low + high)
    tl.store(out_ptr, z)


@triton.jit
def row_sparsity_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    B: tl.int32, S: tl.int32, N: tl.int32,
    std_multiplier: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row (b, s). For each row:
      - First pass: compute sum and sumsq across N
      - Compute mean, std, threshold
      - Second pass: apply gating out = max(0, x - threshold) and store
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S

    # Base pointers for this row
    row_x_ptr = x_ptr + b * (S * N) + s * N
    row_out_ptr = out_ptr + b * (S * N) + s * N

    # First pass: accumulate sum and sumsq in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    n_start = 0
    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        vals = tl.load(row_x_ptr + n_offsets, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sum_val += tl.sum(vals_f32, axis=0)
        sumsq_val += tl.sum(vals_f32 * vals_f32, axis=0)
        n_start += BLOCK_SIZE

    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    # Clamp variance to non-negative to avoid numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: apply gating
    n_start = 0
    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        vals = tl.load(row_x_ptr + n_offsets, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        out = vals_f32 - threshold
        out = tl.maximum(out, 0.0)  # ReLU
        tl.store(row_out_ptr + n_offsets, out, mask=mask)
        n_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return input unchanged (returning x in any dtype is allowed)
        if target_sparsity == 0.0:
            # To match the original behavior, cast to bfloat16 if desired; otherwise return as is.
            # The original returns bfloat16; we cast here to be consistent.
            return x.to(torch.bfloat16)

        # Ensure contiguous and compute in float32
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Allocate output as float32 for kernel writes
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element buffer for std_multiplier (scalar inverse CDF)
        std_multiplier = torch.empty((1,), dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(std_multiplier) in Triton
        # Pass p as Python float; Triton kernel writes into std_multiplier[0]
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, float(std_multiplier[0].item()),
            BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
