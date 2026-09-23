import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr):
    """
    Compute inverse standard normal CDF inv-Phi(p) via bisection.
    Store the result to out_ptr[0] as float32.
    """
    # Work in float32 scalars
    p_val = tl.full((), p, tl.float32)
    lo = tl.full((), -6.0, tl.float32)
    hi = tl.full((), 6.0, tl.float32)

    # Bisection constants
    tol = tl.full((), 1e-7, tl.float32)
    max_iter = tl.full((), 100, tl.int32)

    # Bisection loop
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # erf approximation (Abramowitz & Stegun 7.1.26)
        x = mid
        # t = 1 / (1 + p*|x|), p = 0.3275911
        p_erf = tl.full((), 0.3275911, tl.float32)
        ax = tl.abs(x)
        t = 1.0 / (1.0 + p_erf * ax)
        # Coefficients for erf approximation
        a1 = tl.full((), 0.254829592, tl.float32)
        a2 = tl.full((), -0.284496736, tl.float32)
        a3 = tl.full((), 1.421413741, tl.float32)
        a4 = tl.full((), -1.453152027, tl.float32)
        a5 = tl.full((), 1.061405429, tl.float32)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-x * x)
        phi = 0.5 * (1.0 + erf_approx)  # since we compute with positive x, sign handled via abs
        # For negative x: phi = 0.5 * (1 - erf(x))
        # But since we use |x| and mid may be negative, we need to handle:
        # For our case, p ∈ (0,1), mid typically positive; but make it general:
        # Use sign(mid) for correctness.
        sign = tl.where(mid >= 0.0, 1.0, -1.0)
        phi = 0.5 * (1.0 + sign * erf_approx)  # since erf is odd, erf(-x) = -erf(x)

        # Update interval
        cond = phi > p_val
        lo = tl.where(cond, mid, lo)
        hi = tl.where(cond, hi, mid)

    mid = 0.5 * (lo + hi)
    # Store the result
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK_N: tl.constexpr):
    """
    One Triton program per row (b, s).
    - First pass: compute sum and sum of squares across N.
    - Compute mean and std.
    - Load std_multiplier from device.
    - Compute threshold = mean + std * std_multiplier.
    - Second pass: apply ReLU gating: out = max(0, x - threshold), store float32.
    """
    row_id = tl.program_id(axis=0)
    b = row_id // S
    s = row_id % S

    # Base linear offset for the row
    # With contiguous [B, S, N], row length = N; total elements per row = N
    # Linear offset for start of this row in a flattened [B*S, N] view is row_id * N.
    # But we will index via b, s, n using strides; here x is [B,S,N] contiguous.
    # We'll use b, s, and iterate n. Triton expects flattened indexing, so we compute:
    # For each n, linear offset = ((b*S + s) * N + n). Since axis=0 program is per row,
    # we can compute row base as (b*S + s) * N, but simpler: we will compute per element using b and s.

    # We will load x[b, s, n] directly by computing base = b*S*N + s*N + n.
    # However Triton kernels don't have direct multi-d indexing; we assume x_ptr is a 1D contiguous view.
    # To be safe, we pass x_ptr as 1D contiguous [B*S*N]. Then row base = row_id * N.
    row_base = row_id * N

    # First pass: sum and sumsq
    sum_val = tl.full((), 0.0, tl.float32)
    sumsq_val = tl.full((), 0.0, tl.float32)

    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK_N)
        mask = offs < N
        # Load a chunk of the row, cast to float32
        x_chunk = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_chunk, axis=0)
        sumsq_val += tl.sum(x_chunk * x_chunk, axis=0)
        n += BLOCK_N

    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    # Numerical guard: var should be non-negative; clamp to 0 if negative
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (1-element buffer)
    std_multiplier = tl.load(std_multiplier_ptr)

    threshold = mean + std * std_multiplier

    # Second pass: apply gating and store
    n = 0
    while n < N:
        offs = n + tl.arange(0, BLOCK_N)
        mask = offs < N
        x_chunk = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        y_chunk = x_chunk - threshold
        # ReLU
        y_chunk = tl.maximum(y_chunk, 0.0)
        # Store float32 output
        tl.store(out_ptr + row_base + offs, y_chunk, mask=mask)
        n += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, x, target_sparsity: float):
        """
        Triton-optimized implementation of run(inputs, target_sparsity):
        - If target_sparsity == 0.0, return x unchanged (no sparsity).
        - Else, compute per-(b,s) mean/std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU(x - threshold). Output in bfloat16.
        """
        # No torch ops in forward except for dtype conversions and shape handling.
        # Ensure input is 3D: [B, S, N]
        assert x.dim() == 3, "Input must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, N = x.shape

        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            # Return x in bfloat16 to match original behavior
            return x.to(torch.bfloat16)

        # Make input contiguous and cast to float32 for computation
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare output as float32
        out_f32 = torch.empty(B * S * N, dtype=torch.float32, device=x.device)

        # Device-side scalar for inv-Phi: 1-element tensor
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)

        # Launch compute_invphi_kernel: pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row_sparsity_kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK_N=1024, num_warps=8
        )

        # Reshape output to [B, S, N] and cast to bfloat16 to match original behavior
        out = out_f32.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
