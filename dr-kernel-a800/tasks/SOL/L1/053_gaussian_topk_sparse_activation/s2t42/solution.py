import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr: tl.pointer_type(tl.float32)):
    """
    Compute inverse standard normal CDF for probability p in (0, 1) using bisection
    and Abramowitz & Stegun 7.1.26 erf approximation. Write the result into out_ptr[0].
    """
    # Bisection bounds
    z_low = -6.0
    z_high = 6.0
    # Safety guard: if p <= 0 or p >= 1, clamp
    # (though caller should provide p in (0,1))
    if p <= 0.0:
        tl.store(out_ptr, 0.0)
        return
    if p >= 1.0:
        tl.store(out_ptr, 6.0)  # near infinity, but we won't reach here for valid p
        return

    # Bisection iteration
    eps = 1e-7
    for _ in range(100):
        z_mid = 0.5 * (z_low + z_high)
        # erf(z) approximation (A&S 7.1.26)
        # erf(x) ~ sign(x) * (1 - exp(-x^2) * P(t)), t = 1/(1+p|x|)
        sign = 1.0
        if z_mid < 0.0:
            sign = -1.0
        az = tl.abs(z_mid)
        t = 1.0 / (1.0 + 0.5 * az)
        # Polynomial P(t) coefficients for erf (A&S 7.1.26)
        # P(t) = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
        a1 = 0.0742640736
        a2 = 0.0705266723
        a3 = 0.0730292361
        a4 = 0.0781023579
        a5 = 0.140012201
        poly = a5 * t + a4
        poly = poly * t + a3
        poly = poly * t + a2
        poly = poly * t + a1
        poly = poly * t
        erf_approx = sign * (1.0 - tl.exp(-(az * az)) * poly)
        # Standard normal CDF at z_mid
        phi = 0.5 * (1.0 + erf_approx * 0.7071067811865476)  # 0.70710678... = 1/sqrt(2)
        if phi > p:
            z_low = z_mid
        else:
            z_high = z_mid
    z = 0.5 * (z_low + z_high)
    tl.store(out_ptr, z)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B: tl.int32, S: tl.int32, N: tl.int32, std_multiplier: tl.float32, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per (b, s) row. Two passes over N:
    1) Compute sum and sum of squares to derive mean and std.
    2) Compute threshold = mean + std * std_multiplier and apply ReLU gating: out = max(0, x - threshold).
    All math in float32; inputs should be cast to float32 before launching.
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    base = row_id * N

    # First pass: compute sum and sum of squares (accumulate in float32)
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    # population std (unbiased=False)
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # threshold = mean + std * inv-Phi(sparsity)
    threshold = mean + std * std_multiplier

    # Second pass: apply ReLU gating
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        out = tl.maximum(x - threshold, 0.0)  # ReLU
        tl.store(out_ptr + base + cols, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: input tensor of shape [B, S, N]
        target_sparsity: float in (0, 1)
        Returns: bfloat16 tensor of shape [B, S, N] with sparsity gating applied.
        """
        # No sparsity requested
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and cast to float32 for numeric stability
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Allocate output (float32 for compute, cast later to bfloat16)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Compute std_multiplier = inv-Phi(target_sparsity) using Triton
        std_multiplier = torch.empty((), dtype=torch.float32, device=x_f32.device)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row-wise sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match example behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
