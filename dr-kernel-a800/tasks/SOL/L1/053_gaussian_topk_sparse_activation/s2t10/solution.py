import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, std_multiplier_ptr: tl.pointer_type(tl.float32)):
    """
    Compute inverse standard normal CDF for probability p using bisection.
    Write result to std_multiplier_ptr[0].
    Assumes 0 < p < 1.
    """
    # Bisection over z in [-6, 6]
    lo = -6.0
    hi = 6.0
    # 100 iterations for good precision
    for _ in range(100):
        mid = (lo + hi) * 0.5
        # Abramowitz & Stegun erf approximation
        x = mid
        t = 1.0 / (1.0 + 0.3275911 * tl.abs(x))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        erf_approx = 1.0 - (a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t) * tl.exp(-x * x)
        cdf = 0.5 * (1.0 + erf_approx * tl.where(x >= 0, 1.0, -1.0))
        if cdf > p:
            hi = mid
        else:
            lo = mid
    tl.store(std_multiplier_ptr, (lo + hi) * 0.5)


@triton.jit
def row_sparsity_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    B: tl.int32, S: tl.int32, N: tl.int32,
    std_multiplier_ptr: tl.pointer_type(tl.float32),
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row (b, s). Two passes over N:
    Pass 1: compute mean and std
    Pass 2: compute threshold and write gated output = max(0, x - threshold)
    All arithmetic in float32. Use masked chunked loads/stores.
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S

    base_in = (b * S + s) * N
    base_out = b * S * N + s * N

    # Pass 1: accumulate sum and sum of squares
    total = 0.0
    total2 = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + base_in + offsets, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)

    n = N
    mean = total / n
    var = total2 / n - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (scalar from device)
    std_multiplier = tl.load(std_multiplier_ptr).to(tl.float32)

    threshold = mean + std * std_multiplier

    # Pass 2: apply gating
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + base_in + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base_out + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton implementation of run(inputs, target_sparsity):
        - Compute per-row mean and std in float32.
        - threshold = mean + std * inv-Phi(target_sparsity)
        - output = max(0, input - threshold), returned as bfloat16.
        No torch operations in forward; Triton kernels are used.
        """
        # If no sparsity requested, return input as bfloat16
        if target_sparsity == 0.0:
            return input_tensor.to(torch.bfloat16)

        # Ensure contiguous and cast to float32 for computation
        x = input_tensor.contiguous()
        x_f32 = x.to(torch.float32)
        B, S, N = x_f32.shape
        device = x_f32.device

        # Output buffer (float32 for compute)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=device)

        # Device scalar buffer for std_multiplier
        std_multiplier = torch.empty(1, dtype=torch.float32, device=device)

        # Launch inv-Phi kernel: pass p as Python float
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32,
            B, S, N,
            std_multiplier,  # pointer to scalar
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # Cast to bfloat16 for return
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
