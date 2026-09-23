import torch
import triton
import triton.language as tl


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, p, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute inverse standard normal CDF (quantile) for p in (0,1).
    Uses Abramowitz and Stegun 5.2.23 approximation.
    Writes a single float to z_buf_ptr[0].
    """
    # We implement the piecewise approximation for numerical stability
    # Lower tail
    mask_low = p < p_low
    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    # Upper tail
    mask_high = p > p_high

    # Compute z for each region and store
    # Note: z_buf_ptr is a 1-element buffer; we write scalar result.
    # We'll use a compile-time mask trick via tl.where to avoid branching.

    # Create index 0 for scalar computation
    idx = tl.arange(0, 1)  # dummy to satisfy Triton's expectation
    # Compute q for central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    num_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid
    den_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z_mid = num_mid / den_mid

    # Compute z for lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    num_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = -num_low / den_low

    # Compute z for upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    num_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = num_high / den_high

    # Select the correct z per region
    # Triton scalar: use masks and tl.where
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_high, 0.0)

    # Write to 1-element buffer
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply gating: out = relu(x - (mean + std * z)).
    x_ptr points to flattened [total_rows*K].
    mean_ptr and std_ptr are [total_rows], z_ptr is [1].
    """
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    if row_id >= total_rows:
        return

    start = col_block * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load row base pointers
    x = tl.load(x_ptr + row_id * K + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_ptr)  # scalar
    # Compute
    thr = mean + std * z
    y = x - thr
    y = tl.maximum(y, 0.0)  # relu
    tl.store(out_ptr + row_id * K + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-row mean and std in PyTorch (correct and fast).
        - Compute z = _ndtri(target_sparsity) in PyTorch (accurate inverse CDF).
        - Apply gating with Triton kernel.
        Returns bfloat16 tensor.
        """
        # Ensure input is contiguous and in float32 for compute
        x = x.contiguous()
        x_fp32 = x.to(torch.float32)

        # Compute statistics along last dim (feature dim)
        B, S, K = x_fp32.shape
        total_rows = B * S

        # Per-row mean and std (population std, unbiased=False) as [total_rows]
        # Keepdim to [B, S, 1] for broadcast
        mean = x_fp32.mean(dim=-1, keepdim=True)
        std = x_fp32.std(dim=-1, keepdim=True, unbiased=False)

        # Compute z = _ndtri(target_sparsity) in PyTorch for correctness
        # Using erf-based quantile: z = sqrt(2) * erfinv(2*p - 1)
        # torch.erf is available; erfinv = (erf inverse)
        # Note: p in (0,1); at boundaries, clamp to avoid NaNs
        p_t = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
        # Clamp to (0,1) to avoid NaNs from sqrt of negative due to numerical error
        p_t = torch.clamp(p_t, 0.0, 1.0)
        z = torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=x.device)) * torch.erfinv(2.0 * p_t - 1.0)

        # Prepare output
        out_f32 = torch.empty_like(x_fp32)

        # Launch gating Triton kernel over rows and column tiles
        # Dynamic tuning for BLOCK_SIZE and num_warps
        if K >= 16384:
            block_size_gate = 4096
            num_warps_gate = 4
        elif K >= 8192:
            block_size_gate = 2048
            num_warps_gate = 4
        else:
            block_size_gate = 1024
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_fp32.view(-1), mean.view(-1), std.view(-1), z, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
