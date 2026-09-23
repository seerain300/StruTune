# Triton kernels: compute inv-Phi (quantile) and row sparsity

import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr, iters: tl.constexpr):
    """
    Compute inverse standard normal CDF for probability p using bisection.
    Writes result to out_ptr[0] (float32).
    p: Python float scalar (passed from host). Do not create torch tensors in forward.
    """
    # Lower/upper bounds for bisection
    low = -6.0
    high = 6.0
    # Track result in a 1-element buffer
    result = tl.zeros((), dtype=tl.float32)

    # Bisection iterations
    for _ in range(iters):
        mid = 0.5 * (low + high)
        # erf approximation (Abramowitz & Stegun 7.1.26)
        z = mid * 0.7071067811865476  # 1/sqrt(2)
        az = tl.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * az)
        # Polynomial coefficients
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-(z * z))
        phi = 0.5 * (1.0 + erf_approx)
        # Compare and update bounds
        if phi > p:
            high = mid
        else:
            low = mid
    result = 0.5 * (low + high)
    # Store to 1-element output buffer
    tl.store(out_ptr, result)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s). For each row, compute mean and std across N,
    then apply y = max(0, x - (mean + std * std_multiplier)).
    x_ptr: *f16/f32 (we'll cast to f32 inside)
    out_ptr: *f32
    """
    row_id = tl.program_id(0)  # 0..B*S-1
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N  # index into [B, S, N] flattened

    # First pass: compute sum and sumsq over N
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

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

    # Load std multiplier (inv-Phi(target_sparsity))
    std_multiplier = tl.load(std_multiplier_ptr)

    threshold = mean + std * std_multiplier

    # Second pass: apply ReLU gating
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward:
        - If target_sparsity == 0.0, return inputs.
        - Else, compute per-row mean and std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold). Return in bfloat16.
        """
        # Early exit for no sparsity
        if target_sparsity == 0.0:
            return inputs

        # Expect 3D input [B, S, N]
        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")
        B, S, N = inputs.shape

        # Ensure contiguous and cast to float32 for kernel math
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # 1-element buffer for inv-Phi scalar
        std_multiplier = torch.empty((), dtype=torch.float32, device=x.device)

        # Compute inv-Phi(target_sparsity) using Triton
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier, iters=20)

        # Output tensor in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x.device)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
