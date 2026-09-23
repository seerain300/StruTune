import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, std_multiplier_ptr: tl.pointer(dtype=tl.float32)):
    """
    Compute inverse standard normal CDF for probability p via bisection.
    Write result to std_multiplier_ptr[0] as a 1-element device tensor.
    We work in z ∈ [-6, 6]; bisection converges to high accuracy.
    """
    low = -6.0
    high = 6.0
    tol = 1e-7
    for _ in range(60):
        mid = 0.5 * (low + high)
        # Standard normal CDF using error function approximation (Abramowitz & Stegun 7.1.26)
        sqrt2 = 1.4142135623730951
        z_scaled = mid / sqrt2
        az = tl.abs(z_scaled)
        p_const = 0.3275911
        t = 1.0 / (1.0 + p_const * az)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t) + a4) * t) + a3) * t + a2) * t + a1
        erf_approx = 1.0 - poly * tl.exp(-(az * az))
        # For z_scaled < 0, erf(z_scaled) = -erf(|z_scaled|)
        erf_approx = tl.where(z_scaled < 0.0, -erf_approx, erf_approx)
        cdf = 0.5 * (1.0 + erf_approx)
        if cdf > p:
            high = mid
        else:
            low = mid
    tl.store(std_multiplier_ptr, mid)


@triton.jit
def row_sparsity_kernel(
    x_ptr: tl.pointer(dtype=tl.float32),
    out_ptr: tl.pointer(dtype=tl.float32),
    std_multiplier_ptr: tl.pointer(dtype=tl.float32),
    B: tl.int32, S: tl.int32, N: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    """
    One program per row (b, s). Process the entire row of length N in chunks of BLOCK_SIZE.
    - First pass: accumulate sum and sum of squares to compute mean and std (population std).
    - Second pass: compute threshold = mean + std * std_multiplier and apply ReLU gating.
    """
    row_id = tl.program_id(0)  # 0 .. (B*S - 1)
    b = row_id // S
    s = row_id % S

    # For contiguous [B, S, N], row base offset is (b*S + s) * N.
    base = (b * S + s) * N

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    n_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, n_chunks):
        offs = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_row = tl.load(x_ptr + base + offs, mask=mask, other=tl.float32(0.0))
        sum_val += tl.sum(x_row, axis=0)
        sum_sq += tl.sum(x_row * x_row, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Load std multiplier (1-element tensor)
    std_multiplier = tl.load(std_multiplier_ptr)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: apply gating
    for chunk in range(0, n_chunks):
        offs = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_row = tl.load(x_ptr + base + offs, mask=mask, other=tl.float32(0.0))
        y = x_row - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version:
        - Compute per-row mean and std (float32).
        - threshold = mean + std * inv-Phi(target_sparsity) computed in Triton.
        - output = max(0, x - threshold) per element.
        Return in bfloat16.
        """
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and cast to float32 for computation
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        # Prepare output buffer (flat view for kernel)
        B, S, N = x_f32.shape
        # We will write per-row output to a [B, S, N] buffer by indexing rows explicitly.
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Allocate 1-element device tensor for std_multiplier (scalar result from Triton)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(target_sparsity) via Triton kernel
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, std_multiplier, B, S, N,
            BLOCK_SIZE=1024, num_warps=8
        )

        # Cast to bfloat16 and return
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
