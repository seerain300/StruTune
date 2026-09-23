import torch
import triton
import triton.language as tl


# Constants for Newton's method on Phi(z) = p, where Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
# We'll implement erf approximation within Triton kernel to avoid any torch calls.

@triton.jit
def erf_approx(x):
    # Abramowitz & Stegun (7.1.26) approximation for erf(x):
    # erf(x) ≈ sign(x) * (1 - poly(t) * exp(-x^2)), t = 1 / (1 + p*|x|)
    # Constants
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    x_abs = tl.abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    # Horner's method for polynomial
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    y = 1.0 - poly * tl.exp(-x_abs * x_abs)
    return tl.where(x >= 0, y, -y)


@triton.jit
def phi(z):
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt2 = 1.4142135623730951
    return 0.5 * (1.0 + erf_approx(z / sqrt2))


@triton.jit
def inv_phi_newton(p_val, max_iters: tl.constexpr, out_ptr):
    # Solve for z such that Phi(z) = p using Newton's method:
    # f(z) = Phi(z) - p, f'(z) = phi'(z) = 1 / sqrt(2*pi) * exp(-z^2 / 2) = 0.3989422804 / exp(0.5*z*z)
    # Newton: z_{n+1} = z_n - f(z_n)/f'(z_n) = z_n - (Phi(z_n) - p) / phi'(z_n)
    # Start with an initial guess:
    # For p near 0.5, z ~ (p - 0.5) * sqrt(8/pi)
    # Otherwise, use simple heuristic:
    sqrt2pi = 2.5066282746310002  # sqrt(2*pi)
    sqrt8_over_pi = 2.2567583341729376  # sqrt(8/pi)
    if p_val > 0.5:
        z = (p_val - 0.5) * sqrt8_over_pi
    else:
        z = (0.5 - p_val) * sqrt8_over_pi

    # Constants
    inv_sqrt_2pi = 0.3989422804014327  # 1 / sqrt(2*pi)

    # Newton iterations
    # Note: Triton supports while loops; we use a fixed iteration count as constexpr.
    for _ in range(max_iters):
        phi_z = phi(z)
        # phi'(z) = inv_sqrt_2pi * exp(-0.5*z*z)
        phi_prime = inv_sqrt_2pi * tl.exp(-0.5 * z * z)
        # Newton update
        z = z - (phi_z - p_val) / phi_prime
        # Optional: clamp to avoid extreme values (not necessary if initial guess is good)
        # z = tl.maximum(tl.minimum(z, 6.0), -6.0)

    # Store the result as 1-element tensor
    tl.store(out_ptr, z)


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
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, B, S, H, z_ptr, BLOCK: tl.constexpr):
    # One program per (b, s) row: pid in [0, B*S)
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)

    # Load scalar z (inverse normal CDF) from z_ptr[0]
    z = tl.load(z_ptr)

    # Compute threshold in fp32
    thr = mean + std * z  # fp32

    # Second pass: apply y = max(x - thr, 0) and store as bfloat16
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
        # No sparsity requested
        if target_sparsity == 0.0 or inputs.shape[-1] == 0:
            return inputs

        # Ensure inputs are contiguous
        inputs = inputs.contiguous()

        # Expect 3D input: [B, S, H]
        assert inputs.ndim == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, H = inputs.shape
        device = inputs.device

        # Allocate mean and std buffers in fp32: shape [B, S] => linearized as [B*S]
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Launch compute_mean_std_fp32: grid has one program per (b, s) row
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](inputs, mean_buf, std_buf, B, S, H, BLOCK=1024)

        # Compute z = inverse normal CDF of target_sparsity using Triton kernel (no torch usage)
        # Prepare 1-element fp32 buffer for output
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute z. We pass p_val as a Python float.
        inv_phi_newton[(1,)](target_sparsity, max_iters=6, out_ptr=z_buf)

        # Allocate output buffer in bfloat16
        out = torch.empty_like(inputs, dtype=torch.bfloat16, device=device)

        # Launch apply kernel
        apply_threshold_relu_to_bf16[grid_stats](inputs, out, mean_buf, std_buf, B, S, H, z_buf, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
