import triton
import triton.language as tl


@triton.jit
def inv_phi_kernel(p_ptr: tl.pointer[tl.float32], out_ptr: tl.pointer[tl.float32]):
    # Compute inv-Phi(p) via bisection over z in [-6, 6]
    # p is loaded from p_ptr (1-element device tensor)
    p = tl.load(p_ptr)
    z_low = -6.0
    z_high = 6.0
    eps = 1e-7

    # Standard normal CDF approximation using a closed-form polynomial
    # erf(x) ≈ sign(x) * (1 - t * exp(-x^2) * P(t)), with t = 1 / (1 + p*|x|)
    # We'll use a simple approximation for erf(z/sqrt(2)):
    # erf_approx(z) = sign(z) * (1 - P(|z|))
    # P(x) = (((((a5*x + a4)*x + a3)*x + a2)*x + a1)*x), with x = |z|
    # Coefficients:
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    inv_sqrt2 = 0.7071067811865476

    while True:
        z = 0.5 * (z_low + z_high)
        t = tl.abs(z) * inv_sqrt2
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        # erf(t) ≈ 1 - poly * t * exp(-t^2)
        # sign: +1 for z>=0, -1 for z<0
        sign = 1.0 if z >= 0.0 else -1.0
        erf_t = sign * (1.0 - poly * t * tl.exp(-t * t))
        phi = 0.5 * (1.0 + erf_t)
        if (z >= 0.0 and phi > p) or (z < 0.0 and phi < p):
            z_high = z
        else:
            z_low = z
        if (z_high - z_low) < eps:
            tl.store(out_ptr, z)
            break


@triton.jit
def sparsity_row_kernel(x_ptr, out_ptr, std_multiplier_ptr, B: tl.constexpr, S: tl.constexpr, N: tl.constexpr, p: tl.float32, BLOCK_SIZE: tl.constexpr):
    # One program per (b, s) row
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S

    # Base offsets for this row
    # x_ptr and out_ptr are assumed to be laid out as [B, S, N] contiguous in memory
    # For contiguous [B, S, N], offset = (b*S + s) * N + n
    # We will compute offsets using n = 0..N-1
    # Start: accumulate sum and sumsq in float32
    sum_val = 0.0
    sumsq_val = 0.0

    n = 0
    while n < N:
        offs = (b * S + s) * N + n
        # Load a chunk of size BLOCK_SIZE
        idx = n + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x_chunk = tl.load(x_ptr + offs + idx, mask=mask, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        sum_val += tl.sum(x_chunk, axis=0)
        sumsq_val += tl.sum(x_chunk * x_chunk, axis=0)
        n += BLOCK_SIZE

    mean = sum_val / N
    # population std
    var = sumsq_val / N - mean * mean
    std = tl.sqrt(var)

    # Load inv-Phi(p) scalar
    inv_phi = tl.load(std_multiplier_ptr)
    threshold = mean + std * inv_phi

    # Second pass: write gated output
    n = 0
    while n < N:
        offs = (b * S + s) * N + n
        idx = n + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x_chunk = tl.load(x_ptr + offs + idx, mask=mask, other=0.0).to(tl.float32)
        y = x_chunk - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + offs + idx, y, mask=mask)
        n += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs as-is (original code returns the same dtype)
        if target_sparsity == 0.0:
            return inputs

        # Ensure inputs are contiguous and in float32 for numeric stability
        x_f32 = inputs.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # 1-element buffer for inv-Phi scalar (float32), pass as pointer to kernel
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=x_f32.device)
        std_multiplier = torch.empty((1,), dtype=torch.float32, device=x_f32.device)

        # Launch inv-Phi kernel
        inv_phi_kernel[(1,)](p_tensor, std_multiplier)

        # Allocate output buffer (float32)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        sparsity_row_kernel[grid](
            x_f32, out_f32, std_multiplier, B, S, N, 1024, num_warps=8
        )

        # Return in bfloat16 to match typical behavior in the original code
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
