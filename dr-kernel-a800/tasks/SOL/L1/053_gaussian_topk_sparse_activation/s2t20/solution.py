import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p_scalar: tl.float32, out_ptr):
    """
    Compute inv-Phi(p_scalar) using bisection on z in [-6, 6].
    Write result to out_ptr[0] as a single-element float32 tensor.
    p_scalar: Python float (target_sparsity) passed in as scalar.
    """
    low = -6.0
    high = 6.0
    tol = 1e-7
    iters = 0

    while iters < 25:
        mid = 0.5 * (low + high)
        sqrt2 = 1.4142135623730951
        x = mid / sqrt2
        absx = tl.abs(x)
        p0 = 0.3275911
        t = 1.0 / (1.0 + p0 * absx)
        # Abramowitz & Stegun erf approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-(absx * absx))
        erf_approx = tl.where(x >= 0.0, erf_approx, -erf_approx)
        phi = 0.5 * (1.0 + erf_approx)
        if phi > p_scalar:
            high = mid
        else:
            low = mid
        iters += 1

    mid = 0.5 * (low + high)
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK: tl.constexpr):
    """
    One program per row (b, s):
      - First pass over N: accumulate sum and sum of squares in float32.
      - Compute mean and std (population std).
      - Read inv-Phi(std_multiplier) from std_multiplier_ptr[0].
      - Second pass: compute threshold, apply ReLU gating, write to out_ptr (float32).
    """
    pid = tl.program_id(0)  # 0 <= pid < B*S
    b = pid // S
    s = pid % S
    base = x_ptr + (b * S + s) * N
    out_base = out_ptr + (b * S + s) * N

    # First pass: sum and sum of squares
    sum_val = 0.0
    sumsq_val = 0.0
    idx = 0
    while idx < N:
        offs = idx + tl.arange(0, BLOCK)
        mask = offs < N
        x_vals = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals, axis=0)
        sumsq_val += tl.sum(x_vals * x_vals, axis=0)
        idx += BLOCK

    n_f = N
    mean = sum_val / n_f
    var = sumsq_val / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # numerical safety
    std = tl.sqrt(var)

    # Read inv-Phi multiplier
    std_multiplier = tl.load(std_multiplier_ptr)

    threshold = mean + std * std_multiplier

    # Second pass: apply gate and store
    idx = 0
    while idx < N:
        offs = idx + tl.arange(0, BLOCK)
        mask = offs < N
        x_vals = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        out_vals = x_vals - threshold
        out_vals = tl.maximum(out_vals, 0.0)  # ReLU
        tl.store(out_base + offs, out_vals, mask=mask)
        idx += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        # If no sparsity, return x (cast to bfloat16 to match original behavior)
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure input is contiguous and compute in float32
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        B, S, N = x_f32.shape
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element tensor for inv-Phi scalar
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch compute_invphi_kernel with Python float p
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK=1024, num_warps=4
        )

        # Return in bfloat16 to match the original example
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
