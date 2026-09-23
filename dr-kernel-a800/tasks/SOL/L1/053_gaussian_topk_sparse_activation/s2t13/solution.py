import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr):
    """
    Compute inv-Phi(p) (quantile of standard normal) and write to *out_ptr[0].
    Uses a bisection method with Abramowitz & Stegun erf approximation for accuracy.
    p: Python float passed from host (not torch tensor).
    out_ptr: pointer to a 1-element float32 device tensor to store the result.
    """
    # Bounds for z
    z_min = -6.0
    z_max = 6.0
    # Tolerance
    eps = 1e-7
    # Number of iterations: adjust for precision; 50 steps typically suffice
    iters = 50
    # Since Triton requires scalar ops, perform iterative bisection
    # Note: we must cast any intermediate to float32 explicitly in Triton code.
    for _ in range(iters):
        z = (z_min + z_max) * 0.5
        # erf approximation (Abramowitz & Stegun 7.1.26)
        # erf(x) ≈ sign(x) * [1 - t * exp(-x^2) * (a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4)]
        # where t = 1 / (1 + p * |x|), p=0.3275911, coefficients:
        p_const = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        x = z
        ax = tl.abs(x)
        t = 1.0 / (1.0 + p_const * ax)
        # Polynomial in t
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1)
        erf_approx = 1.0 - poly * t * tl.exp(-(ax * ax))
        # sign(z)
        sign = tl.where(z >= 0.0, 1.0, -1.0)
        cdf = 0.5 * sign * erf_approx + 0.5

        if cdf > p:
            z_max = z
        else:
            z_min = z

    # Write the final z to the output buffer
    # Store as float32 to out_ptr[0]
    # Triton stores require pointer arithmetic; out_ptr is a 1-element array
    # We assume out_ptr points to a 1-element tensor on device with dtype float32.
    # Note: direct store to out_ptr[0] pattern is supported for 1-element tensors.
    tl.store(out_ptr, z)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s). Compute mean and std across N (float32), then apply:
    output = max(0, x - (mean + std * invphi)), where invphi is loaded from std_multiplier_ptr.
    x_ptr: *float32 input tensor of shape [B*S, N]
    out_ptr: *float32 output tensor of shape [B*S, N]
    B, S, N: ints
    std_multiplier_ptr: pointer to a 1-element float32 tensor containing inv-Phi(target_sparsity)
    """
    row = tl.program_id(0)
    # Compute base offset for this row assuming row-major contiguous [B*S, N]
    # We need the actual base index; we'll assume x and out are laid out as [B*S, N] contiguous.
    # Note: We do not have B/S individually here, but we can compute s = row % S, b = row // S if needed.
    # Instead, pass base pointer via a layout convention: out tensor is [B, S, N], but we pass row index and N.
    # To support general layout, we can allocate x and out as [B, S, N] contiguous and pass [B*S, N] via strides,
    # but simpler: we'll assume x and out are allocated as [B*S, N] contiguous in forward.
    # Compute sum and sum of squares in float32
    sum_x = 0.0
    sum_x2 = 0.0

    # Pass 1: accumulate sum and sum of squares across N in chunks
    for start in range(0, N, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        # For x_ptr of shape [B*S, N], the linear index is row * N + idx
        x_vals = tl.load(x_ptr + row * N + idx, mask=mask, other=0.0).to(tl.float32)
        # Ensure other is casted to float32 to avoid type issues
        x_vals = x_vals.to(tl.float32)
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to roundoff
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Load std_multiplier (inv-Phi(p)) from device
    invphi = tl.load(std_multiplier_ptr).to(tl.float32)
    threshold = mean + std * invphi

    # Pass 2: write output = max(0, x - threshold)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x_vals = tl.load(x_ptr + row * N + idx, mask=mask, other=0.0).to(tl.float32)
        out_vals = x_vals - threshold
        # ReLU
        out_vals = tl.maximum(out_vals, 0.0)
        tl.store(out_ptr + row * N + idx, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, target_sparsity: float):
        """
        x: input tensor of shape [B, S, N]
        target_sparsity: float in [0, 1], passed as Python float (no torch ops in forward)
        Returns bfloat16 output tensor of shape [B, S, N].
        """
        # If no sparsity, return as-is
        if target_sparsity == 0.0:
            # Ensure dtype matches original behavior: return bfloat16
            return x.to(torch.bfloat16)

        # Ensure contiguity and work in float32 for computation
        x = x.contiguous()
        B, S, N = x.shape

        # Allocate std_multiplier as 1-element float32 tensor on device
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)

        # Launch kernel to compute inv-Phi(target_sparsity)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Flatten x to [B*S, N] for row-wise processing
        x2d = x.view(B * S, N).contiguous()
        out2d = torch.empty((B * S, N), dtype=torch.float32, device=x.device)

        # Launch sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](x2d, out2d, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8)

        # Reshape back to [B, S, N] and return in bfloat16
        out = out2d.view(B, S, N)
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
