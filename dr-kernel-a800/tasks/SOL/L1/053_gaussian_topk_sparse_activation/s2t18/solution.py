import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr):
    """
    Compute inverse standard normal CDF for p in (0, 1).
    We use bisection over z in [-6, 6] and evaluate standard normal CDF via an approximation
    that only uses exp, add, mul, which are supported by Triton.

    p: scalar float (Python float passed to kernel)
    out_ptr: pointer to a single float32 element where result is stored
    """
    # Bisection bounds
    low = -6.0
    high = 6.0
    # Tolerance
    tol = 1e-6
    # Number of iterations
    iters = 100
    # Since Triton runs kernels in parallel, we use per-thread scalar state.
    # We'll perform the bisection in a fixed iteration loop.
    for _ in range(iters):
        z = (low + high) * 0.5
        # Compute CDF(z): 0.5 * (1 + erf(z / sqrt(2)))
        # Implement erf approximation (Abramowitz & Stegun 7.1.26)
        x = z / 1.4142135623730951  # 1/sqrt(2)
        # erf approximation constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        t = 1.0 / (1.0 + 0.3275911 * tl.abs(x))
        # Polynomial in t
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_x = 1.0 - poly * tl.exp(-x * x)
        cdf = 0.5 * (1.0 + erf_x)
        if cdf > p:
            high = z
        else:
            low = z
    # Store result
    tl.store(out_ptr, (low + high) * 0.5)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, std_ptr, B, S, N, strideB, strideS, strideN, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s).
    x_ptr: *float32 input
    out_ptr: *float32 output
    std_ptr: *float32 scalar std_multiplier
    B, S, N: int sizes
    strideB, strideS, strideN: int strides for x/out
    """
    pid = tl.program_id(axis=0)  # program id corresponds to row index in [0, B*S)
    b = pid // S
    s = pid % S

    # First pass: compute mean and population std across N
    sum_x = 0.0
    sum_x2 = 0.0
    # Iterate over N in chunks of BLOCK_SIZE
    for base in range(0, N, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Compute linear indices: idx = b*strideB + s*strideS + offs*strideN
        idx = b * strideB + s * strideS + offs * strideN
        vals = tl.load(x_ptr + idx, mask=mask, other=tl.float32(0.0))
        # Accumulate sums
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)

    n_float = tl.float32(N)
    mean = sum_x / n_float
    # Population variance (unbiased=False)
    var = sum_x2 / n_float - mean * mean
    # Clamp var to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (scalar)
    std_multiplier = tl.load(std_ptr)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: apply ReLU gating: out = max(0, x - threshold)
    for base in range(0, N, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        idx = b * strideB + s * strideS + offs * strideN
        x_vals = tl.load(x_ptr + idx, mask=mask, other=tl.float32(0.0))
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        x: input tensor of shape [B, S, N], any floating dtype supported by Triton
        target_sparsity: float in (0, 1), inverse normal quantile
        Returns: bfloat16 tensor of shape [B, S, N] with gated activations.
        """
        if target_sparsity == 0.0:
            # No gating
            return x

        # Ensure inputs are contiguous and cast to float32 for numeric stability
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape
        device = x_f32.device

        # Output buffer (float32)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=device)

        # Allocate std_multiplier as 1-element tensor on device
        std_multiplier = torch.empty(1, dtype=torch.float32, device=device)

        # Launch Triton kernel to compute inverse normal CDF for p = target_sparsity
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Strides for last-dimension access
        strideB, strideS, strideN = x_f32.stride()

        # Launch Triton kernel to perform sparsity gating, one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, std_multiplier, B, S, N, strideB, strideS, strideN,
            BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
