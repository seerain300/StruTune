import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF (quantile) for probability p using bisection.
    Writes the result into out_ptr[0] as float32.
    Search over z in [-6, 6]; iterate until convergence.
    """
    z_low = tl.float32(-6.0)
    z_high = tl.float32(6.0)
    tol = tl.float32(1e-7)
    i = 0
    while i < 100:
        z_mid = 0.5 * (z_low + z_high)
        inv_sqrt2 = tl.float32(0.7071067811865476)  # 1/sqrt(2)
        # Standard normal CDF: 0.5 * (1 + erf(z / sqrt(2)))
        cdf = 0.5 * (1.0 + tl.math.erf(z_mid * inv_sqrt2))
        if cdf < p:
            z_low = z_mid
        else:
            z_high = z_mid
        i += 1
    # Write scalar result to out_ptr[0]
    tl.store(out_ptr + 0, z_mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, std_multiplier_ptr, B: tl.int32, S: tl.int32, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (b, s). Computes:
      - mean and std for that row across N (population std)
      - threshold = mean + std * std_multiplier
      - y = max(0, x - threshold)
    Writes float32 output.
    """
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S

    # Base offset for this row: assuming x is [B, S, N] contiguous, each row has N elements
    base = row_id * N

    # First pass: compute sum and sum of squares across N
    sum_val = tl.float32(0.0)
    sum_sq = tl.float32(0.0)
    start = tl.int32(0)
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=tl.float32(0.0))
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    N_f = tl.float32(N)
    mean = sum_val / N_f
    # population variance: E[x^2] - (E[x])^2
    var = sum_sq / N_f - mean * mean
    var = tl.maximum(var, tl.float32(0.0))  # guard against tiny negative due to rounding
    std = tl.sqrt(var)

    # Load std multiplier (scalar) from device
    std_multiplier = tl.load(std_multiplier_ptr)

    # threshold = mean + std * std_multiplier
    threshold = mean + std * std_multiplier

    # Second pass: apply gating and write output
    start = tl.int32(0)
    while start < N:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=tl.float32(0.0))
        y = x - threshold
        y = tl.maximum(y, tl.float32(0.0))  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-optimized implementation of the Gaussian-based top-k sparse activation:
        - Computes per-row mean and std.
        - Computes threshold = mean + std * inv-Phi(target_sparsity).
        - Outputs ReLU(x - threshold).
        No torch operations in forward (except shape metadata and dtype cast at return).
        """
        # Expect input tensor as args[0]; target_sparsity as args[1] as a Python float
        x = args[0]
        if len(args) < 2:
            # If target_sparsity not provided, default to 0.0 (no sparsity)
            target_sparsity = 0.0
        else:
            target_sparsity = float(args[1])

        # If no sparsity, return input unchanged (convert to bfloat16 to match original example behavior)
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous and compute in float32 (no torch tensor creation in forward)
        x_f32 = x.contiguous().to(torch.float32)

        B, S, N = x_f32.shape
        device = x_f32.device

        # Output buffer (float32 for computation)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=device)

        # Scalar std_multiplier buffer (1 element) on device
        std_multiplier = torch.empty(1, dtype=torch.float32, device=device)

        # Launch kernel to compute inverse normal CDF (scalar)
        compute_invphi_kernel[(1,)](target_sparsity, std_multiplier)

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, std_multiplier, B, S, N, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
