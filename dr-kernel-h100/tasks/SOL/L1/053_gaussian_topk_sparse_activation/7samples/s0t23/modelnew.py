import torch
import triton
import triton.language as tl


@triton.jit
def compute_ndtri_kernel(z_buf, p: tl.float32, p_low: tl.float32):
    """
    Compute inverse standard normal CDF at p using Abramowitz & Stegun 5.2.23 approximation.
    z_buf is a 1-element buffer to store result (float32).
    """
    # Constants (A&S 5.2.23)
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00;
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01;
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00;
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00;

    p_low_f = p_low
    # Region masks
    mask_low = p < p_low_f
    mask_mid = (p >= p_low_f) & (p <= (1.0 - p_low_f))
    mask_high = p > (1.0 - p_low_f)

    # Default result
    z = 0.0

    # Lower region
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        z = poly / poly2

    # Central region
    if mask_mid:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly / poly2

    # Upper region
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        z = -poly / poly2

    # Store to 1-element buffer
    tl.store(z_buf, z)


@triton.jit
def apply_gating_2d_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr,
                           total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Gating kernel: out = relu(x - (mean + std * z))
    x_ptr, out_ptr are flattened [total_rows*K], mean_ptr, std_ptr are [total_rows],
    z_ptr is [1] scalar threshold factor. All computations in float32.
    """
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    if row_id >= total_rows:
        return

    # Load mean and std for this row
    mean_row = tl.load(mean_ptr + row_id)
    std_row = tl.load(std_ptr + row_id)
    z_val = tl.load(z_ptr)  # scalar threshold factor

    col_start = tile_id * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < K
    row_base = row_id * K

    # Load x, upcast to fp32
    x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    threshold = mean_row + std_row * z_val
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store result
    tl.store(out_ptr + row_base + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run(inputs, target_sparsity):
        - Compute per-row mean and std using PyTorch in fp32 (robust and accurate).
        - Compute z = ndtri(target_sparsity) in Triton.
        - Apply gating in Triton: out = relu(x - (mean + std * z)).
        - Return bfloat16 output.
        """
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute stats in fp32
        x_contig = x.contiguous()
        K = x_contig.shape[-1]
        B = x_contig.shape[0]
        S = x_contig.shape[1]
        total_rows = B * S

        x_f32 = x_contig.to(torch.float32)
        # Compute mean and std along last dim (population std, unbiased=False)
        mean = torch.mean(x_f32, dim=-1, keepdim=True)  # [B, S, 1]
        std = torch.std(x_f32, dim=-1, keepdim=True, unbiased=False)  # [B, S, 1]

        # Allocate 1-element buffer for z and compute inverse CDF in Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        # Pass only required arguments to Triton kernel
        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity), 0.02425,  # p and p_low; no extra args
            num_warps=1, num_stages=1
        )

        # Prepare input and output for gating in Triton
        out_f32 = torch.empty_like(x_f32)

        # Dynamic tuning for gating kernel based on K
        if K >= 8192:
            block_size_gate = 4096
            num_warps_gate = 8
        else:
            block_size_gate = 2048
            num_warps_gate = 4

        grid_gate = (total_rows, triton.cdiv(K, block_size_gate))
        apply_gating_2d_kernel[grid_gate](
            x_f32.view(-1), mean.view(-1), std.view(-1), z_buf, out_f32.view(-1),
            total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate,
            num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)