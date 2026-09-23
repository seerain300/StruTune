import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF (quantile) for probability p in (0, 1).
    Uses bisection on z in [-6, 6] with high-accuracy erf approximation (Abramowitz & Stegun 7.1.26).
    Writes result to out_ptr (1-element float32 tensor).
    """
    # 1-element output buffer
    out_ptr = out_ptr  # pointer to single float32

    # bisection bounds
    z_low = -6.0
    z_high = 6.0

    # ERF approximation constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # bisection iterations
    for _ in range(25):
        z = 0.5 * (z_low + z_high)
        # erf(z) approximation
        x = z if z >= 0.0 else -z
        t = 1.0 / (1.0 + a1 * x)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-(x * x))
        erf_approx = erf_approx if z >= 0.0 else 2.0 * (0.5 - poly * tl.exp(-(x * x)))
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        phi = 0.5 * (1.0 + erf_approx * 0.7071067811865476)  # 1/sqrt(2)

        if phi > p:
            z_high = z
        else:
            z_low = z

    # write result
    tl.store(out_ptr, z_low)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, mean_ptr, std_ptr, std_multiplier_ptr,
                         B: tl.int32, S: tl.int32, N: tl.int32,
                         BLOCK_SIZE: tl.constexpr):
    """
    One program per (b, s) row. Computes mean and std across N, then applies
    elementwise ReLU gating with threshold = mean + std * std_multiplier.
    Stores output in out_ptr (float32). x_ptr is assumed to be float32.
    """
    pid = tl.program_id(0)  # 0 .. B*S - 1
    s = pid % S
    b = pid // S

    # Row base pointers (contiguous layout: [B, S, N] => index = b*S*N + s*N + n)
    row_base_x = x_ptr + b * S * N + s * N
    row_base_out = out_ptr + b * S * N + s * N

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        vals = tl.load(row_base_x + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    # Compute mean and std (population std)
    N_f = tl.float32(N)
    mean = sum_val / N_f
    var = sum_sq / N_f - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (scalar) from device
    std_multiplier = tl.load(std_multiplier_ptr).to(tl.float32)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: apply ReLU gating
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_base_x + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(row_base_out + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized forward:
        - If target_sparsity == 0.0, return x unchanged.
        - Else compute mean/std per row, compute inv-Phi on device, then apply ReLU gating.
        - Return output in bfloat16.
        """
        # If no sparsity requested, return as-is
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32 (to match original behavior)
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape

        # Allocate 1-element device buffer for std_multiplier (scalar output of inv-Phi)
        std_multiplier = torch.empty((), dtype=torch.float32, device=x_f32.device)

        # Launch inv-Phi kernel: pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Allocate output in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, None, None, std_multiplier,
            B, S, N, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
