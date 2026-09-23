import torch
import triton
import triton.language as tl


@triton.jit
def compute_mean_std_fp32(in_ptr, mean_ptr, std_ptr, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # First pass: accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0

    offsets = tl.arange(0, BLOCK)
    i = 0
    base = b * S * H + s * H
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
    var = sum_sq / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Store per-row mean and std
    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


@triton.jit
def compute_icdf_scalar_newton(out_ptr, p_val, BLOCK: tl.constexpr):
    # Compute inverse standard normal CDF for p_val via Newton's method:
    # Solve f(z) = 0.5 * (1 + erf(z / sqrt(2))) - p_val = 0
    # f'(z) = (2/sqrt(pi)) * exp(-z^2 / 2)
    sqrt2 = 1.4142135623730951
    two_over_sqrt_pi = 2.0 / 8.853645121445618  # 1/sqrt(pi/2)

    # Initial guess: z0 = sign(p-0.5) * 5.0 * (p-0.5)
    # Note: Triton doesn't provide erf, but we can loop and update out_ptr
    z = tl.load(out_ptr)  # initialize z; will be updated in loop
    # Since Triton JIT cannot use arbitrary control flow outside loops, we implement fixed iterations.
    # Use 6 iterations for good accuracy.
    # We need to compute erf(z/sqrt(2)); implement a simple approximation using Taylor series up to n=6 for |z|<=2
    # and use erf(z) ≈ sign(z) * (1 - exp(-z^2) * P(t)) for better accuracy. However, Triton lacks some math fns.
    # To keep it simple and accurate enough, we use a standard approximation for erf and Newton steps.
    # Here we approximate erf(u) with a well-known formula (Abramowitz-Stegun 7.1.26):
    # erf(u) ≈ 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-u^2), where t = 1 / (1 + p u), p = 0.3275911
    # We'll implement this per iteration to get a good erf(z/sqrt(2)).

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Newton iterations: update z for 6 steps
    # We'll recompute erf each time; Triton supports while loops.
    # Note: Triton JIT does not allow while with dynamic stop; use a fixed iteration count.
    # Implement 6 iterations with a for loop. Triton supports for loops with constexpr range.
    for _ in range(6):
        u = z / sqrt2
        au = tl.abs(u)
        t = 1.0 / (1.0 + p * au)
        # Horner evaluation of poly = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = 1.0 - poly * tl.exp(-(au * au))  # erf(u) with sign u>0; using abs, sign handled later
        sign = tl.where(u >= 0.0, 1.0, -1.0)
        erf_approx = sign * erf_approx
        phi = 0.5 * (1.0 + erf_approx) - p_val
        # f'(z) = (2/sqrt(pi)) * exp(-z^2 / 2)
        df = two_over_sqrt_pi * tl.exp(-0.5 * z * z)
        # Newton update: z_next = z - phi / df
        z = z - phi / df
        tl.store(out_ptr, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)

    # Load scalar z (inverse CDF) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold for this row (fp32 scalar)
    thr = mean + std * z  # fp32

    # Second pass: apply y = max(x - thr, 0) to each feature
    offsets = tl.arange(0, BLOCK)
    base = b * S * H + s * H
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, just return inputs
        if target_sparsity == 0.0:
            return inputs

        # Ensure 3D input: [B, S, H]
        assert inputs.ndim == 3, "inputs must be 3D: [B, S, H]"
        inputs = inputs.contiguous()
        B, S, H = inputs.shape
        device = inputs.device

        # Allocate buffers for mean and std (fp32)
        mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Launch kernel to compute mean and std per (b, s) row
        BLOCK = 1024
        grid = (B * S,)
        compute_mean_std_fp32[grid](
            inputs, mean, std, B, S, H,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute inverse normal CDF for target_sparsity using Triton (Newton method)
        z_out = torch.empty((1,), dtype=torch.float32, device=device)
        # Use p_val directly; the kernel will run several iterations to refine z
        grid_icdf = (1,)
        compute_icdf_scalar_newton[grid_icdf](
            z_out, float(target_sparsity),
            BLOCK=1,  # scalar kernel computes into z_out[0]
            num_warps=1,
        )

        # Prepare output (bfloat16) and apply threshold + ReLU
        out = torch.empty((B, S, H), dtype=torch.bfloat16, device=device)
        apply_threshold_relu_to_bf16[grid](
            inputs, out, mean, std, B, S, H, z_out,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
