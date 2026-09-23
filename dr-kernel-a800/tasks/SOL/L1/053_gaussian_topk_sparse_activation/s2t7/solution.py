import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF for probability p.
    Uses bisection on z in [-6, 6] and an erf approximation (Abramowitz & Stegun 7.1.26).
    Writes result to out_ptr[0] as float32.
    """
    # Bisection bounds
    low = -6.0
    high = 6.0
    result = tl.zeros((), dtype=tl.float32)

    # Perform bisection iterations
    for _ in range(20):
        mid = 0.5 * (low + high)
        # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
        sqrt2 = 1.4142135623730951
        z = mid / sqrt2
        # erf approximation (Abramowitz & Stegun 7.1.26)
        ax = tl.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-(ax * ax))
        cdf = 0.5 * (1.0 + erf_approx)
        # Update bounds
        if cdf > p:
            high = mid
        else:
            low = mid
        result = 0.5 * (low + high)
    # Store result to out_ptr[0]
    tl.store(out_ptr, result)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B: tl.int32, S: tl.int32, N: tl.int32, std_multiplier: tl.float32, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per row (pid maps to a row). Computes:
      - mean and std over N elements in x_ptr for the row.
      - threshold = mean + std * std_multiplier
      - output = max(0, x - threshold)
    x_ptr is a flat pointer to [B*S, N]; each program uses base = pid * N.
    """
    pid = tl.program_id(0)
    base = pid * N

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # guard against small negative due to roundoff
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: elementwise gating
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
        Triton-only forward:
        - If target_sparsity == 0.0, return inputs unchanged.
        - Else compute per-row mean and std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold). Return in bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")

        B, S, N = inputs.shape

        # Ensure contiguous and use float32 for kernel math
        x = inputs.contiguous().to(torch.float32)

        # Output tensor in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x.device)

        # Compute inv-Phi(target_sparsity) on device; pass std_multiplier as a Python float to the kernel
        std_multiplier = 0.0  # placeholder, will be computed by kernel; do not create torch tensor in forward
        # We cannot create a torch tensor for std_multiplier here because that would involve torch.* in forward.
        # Instead, we pass the Python float to the kernel below and compute it in-kernel using out_ptr.
        # To do this, we allocate a 1-element tensor on device and pass its pointer to compute_invphi_kernel.
        std_multiplier_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier_buf)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x, out_f32, B, S, N, std_multiplier_buf[0], BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
