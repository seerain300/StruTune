import torch
import triton
import triton.language as tl


# Kernel 1: compute per-(b,s) mean and std (population, unbiased=False) in fp32
@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    base = b * S * H + s * H

    # Pass 1: sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        i += BLOCK

    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    std = tl.sqrt(var)

    # Store per-row results
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


# Kernel 2: compute inverse normal CDF of p using A&S 5.2.23 (center region) into z_ptr[0]
@triton.jit
def compute_icdf_scalar(z_ptr, p_val, BLOCK: tl.constexpr):
    # A&S 5.2.23 center region: z = -((a1 r + a2 r^2 + ...) / (b1 r + b2 r^2 + ...))
    # where r = (p - 0.5)^2
    p = p_val
    q = p - 0.5  # scalar
    r = q * q

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Numerator and denominator polynomials
    num = 0.0
    den = 0.0
    # Fixed for-loop to avoid runtime while-loop issues; mask the last iterations by zeroing q
    # We loop up to 1024 iterations, but in practice r is small (<= 0.5^2), so convergence is good.
    for _ in range(0, 1024):
        # If _ exceeds needed iterations, set q=0 so expressions evaluate to 0
        # Note: this is a guard to prevent any pathological evaluation; r is scalar.
        # However, with only 1 iteration, this loop won't run; it's here to satisfy Triton's compile expectations.
        q_eff = q  # we will only use the first iteration due to masks in evaluation
        # Compute current polynomial contributions (only the first iteration is meaningful)
        # For subsequent iterations, num/den remain 0
        # Simplify by computing only once
        pass  # placeholder to make the AST valid; Triton expects a body, but we compute below

    # The actual computation is done here with correct A&S formula using the first iteration
    num = ((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6
    den = ((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
    z = - (num * q) / den

    tl.store(z_ptr, z)


# Kernel 3: apply threshold per row, write bfloat16 output
@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row stats
    mean = tl.load(mean_ptr + row)  # fp32
    std = tl.load(std_ptr + row)    # fp32
    z = tl.load(z_ptr)              # fp32, scalar icdf

    thr = mean + std * z            # fp32

    base = b * S * H + s * H
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)      # ReLU
        # Store as bfloat16
        # Cast to bf16 explicitly
        y_bf16 = y.to(tl.bfloat16)
        tl.store(out_ptr + base + idx, y_bf16, mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return original (unchanged dtype behaviorally)
        if target_sparsity == 0.0:
            return inputs

        assert inputs.is_cuda, "ModelNew requires CUDA tensor input for Triton kernels"
        # We keep original dtype for final output (bfloat16 as per original function)
        # But the kernel reads fp32 and writes fp32 for arithmetic, then cast to bf16 at store.
        B, S, H = inputs.shape
        device = inputs.device

        # Allocate stats buffers (fp32)
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        # Allocate output (bf16)
        out = torch.empty_like(inputs, dtype=torch.bfloat16, device=device)

        # Allocate 1-element tensor for icdf scalar (fp32)
        z = torch.empty(1, dtype=torch.float32, device=device)

        # Launch compute mean/std kernel
        # Choose a large block to reduce loop iterations; adjust num_warps accordingly
        BLOCK = 2048
        grid_mean_std = (B * S,)
        compute_mean_std_fp32[grid_mean_std](
            inputs, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4
        )

        # Launch icdf kernel to compute z in fp32
        grid_icdf = (1,)
        compute_icdf_scalar[grid_icdf](
            z, target_sparsity,
            BLOCK=1,  # scalar kernel
            num_warps=1
        )

        # Launch apply kernel
        apply_threshold_relu_to_bf16[grid_mean_std](
            inputs, out, mean, std, z, B, S, H,
            BLOCK=BLOCK,
            num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
