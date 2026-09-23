import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr, iters: tl.constexpr):
    """
    Compute inverse standard normal CDF for probability p (0 < p < 1) using bisection.
    Writes result to out_ptr[0] as float32.
    """
    low = -6.0
    high = 6.0
    # Seed mid as 0.0; bisection will refine it
    mid = 0.0

    for _ in range(iters):
        mid = 0.5 * (low + high)
        # Normal CDF: 0.5 * (1 + erf(mid / sqrt(2)))
        sqrt2 = 1.4142135623730951
        z = mid / sqrt2
        # erf approximation (Abramowitz & Stegun 7.1.26)
        sign = tl.where(z >= 0.0, 1.0, -1.0)
        az = tl.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * az)
        # Polynomial coefficients
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
        cdf = 0.5 * (1.0 + erf_approx)
        # Decide direction
        go_low = cdf > p
        if go_low:
            low = mid
        else:
            high = mid
    # Store result
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s).
    Computes per-row mean and std, then applies ReLU gating with threshold =
      mean + std * std_multiplier_ptr[0].
    x_ptr: *float (input, any float dtype, will cast to float32 inside)
    out_ptr: *float (output, float32)
    """
    pid = tl.program_id(0)  # linear program id
    base = pid * N
    # First pass: compute sum and sum of squares in float32
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
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to roundoff
    std = tl.sqrt(var)

    # Load scalar std_multiplier (1-element tensor)
    std_multiplier = tl.load(std_multiplier_ptr)  # float32 scalar

    threshold = mean + std * std_multiplier

    # Second pass: apply ReLU gate
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU: max(0, y)
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward:
        - If target_sparsity == 0.0, return inputs unchanged.
        - Else compute per-row mean and std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold). Return in bfloat16.
        """
        # No torch compute allowed here; inputs are expected to be provided by the harness.
        # We assume inputs is a torch.Tensor, but we avoid any torch math.
        if target_sparsity == 0.0:
            return inputs

        # Ensure we have a 3D input [B, S, N]
        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")

        B, S, N = inputs.shape

        # Make input contiguous and compute in float32 inside kernel
        x = inputs.contiguous()

        # Output tensor in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x.device)

        # Device-side scalar for inv-Phi(target_sparsity)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)
        # Launch inv-phi kernel with scalar p (Python float)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier, iters=20)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
