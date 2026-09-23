import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and std across K for each row flattened to [total_rows].
    x_ptr points to a contiguous [total_rows, K] view.
    """
    row_id = tl.program_id(0)
    # fp32 accumulators
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Loop over K in tiles
    for start in range(0, K, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        # pointer to the start of the row + idx
        row_offset = row_id * K
        vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        acc_sum += tl.sum(vals, axis=0)
        acc_sumsq += tl.sum(vals * vals, axis=0)

    mean = acc_sum / K
    # population std (unbiased=False), matching torch.std(..., unbiased=False)
    var = acc_sumsq / K - mean * mean
    std = tl.sqrt(var)

    # Store results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity,  # pass target_sparsity as float (host scalar)
                         a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low,
                         BLOCK_SIZE: tl.constexpr):
    """
    Compute z = _ndtri(target_sparsity) via Abramowitz & Stegun 5.2.23 approximation.
    Store result in z_buf_ptr[0] as float32. No torch.tensor usage in host.
    """
    # Only one program instance needed; we fill z_buf_ptr[0]
    p_low = p_low
    p = target_sparsity  # host passes float

    # Masks for regions
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Initialize z
    z = 0.0

    # Lower region
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = poly / poly2

    # Central region
    if mask_mid:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly / poly2

    # Upper region
    if mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        z = -poly / poly2

    # Store result
    tl.store(z_buf_ptr, z)


@triton.jit
def apply_threshold_kernel(x_ptr, mean_ptr, std_ptr, z_buf_ptr, thresh_ptr,
                            total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row, compute threshold = mean + std * z and write to thresh_ptr[row*K : (row+1)*K].
    """
    row_id = tl.program_id(0)
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    z = tl.load(z_buf_ptr)  # scalar z
    threshold = mean + std * z

    # Write threshold across K elements for this row
    for start in range(0, K, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        row_offset = row_id * K
        vals = tl.full([BLOCK_SIZE], threshold, tl.float32)
        tl.store(thresh_ptr + row_offset + idx, vals, mask=mask)


@triton.jit
def gating_relu_kernel(x_ptr, thresh_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise: out = max(x - thresh, 0). thresh is per-row buffer of length total_rows*K.
    """
    row_id = tl.program_id(0)
    for start in range(0, K, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        row_offset = row_id * K
        x_vals = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        th = tl.load(thresh_ptr + row_offset + idx, mask=mask, other=0.0)
        # Ensure compute in fp32
        x_vals = x_vals.to(tl.float32)
        th = th.to(tl.float32)
        out_vals = x_vals - th
        out_vals = tl.maximum(out_vals, 0.0)
        tl.store(out_ptr + row_offset + idx, out_vals, mask=mask)


def _choose_params(K: int):
    # Dynamic tiling heuristic
    if K >= 8192:
        return 4096, 8, 2
    else:
        return 2048, 4, 2


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation:
        - Compute per-row mean and std
        - Compute z = _ndtri(target_sparsity) in Triton
        - Compute threshold = mean + std * z in Triton
        - Apply gating out = relu(x - threshold) in Triton
        - Return bfloat16
        """
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "ModelNew requires CUDA tensors"
        x = inputs.contiguous()
        B, S, K = x.shape
        total_rows = B * S

        # Flatten to [total_rows, K] for simple indexing
        x_flat = x.view(total_rows, K)

        # Compute stats in fp32
        x_flat_f32 = x_flat.to(torch.float32)
        mean = torch.empty(total_rows, dtype=torch.float32, device=x.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=x.device)

        # Launch reduction kernel to compute per-row mean and std
        BLOCK_SIZE_STATS = 2048
        grid_stats = (total_rows,)
        compute_row_stats_kernel[grid_stats](
            x_flat_f32, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_STATS,
            num_warps=4, num_stages=2
        )

        # Compute z = _ndtri(target_sparsity) in Triton without torch.tensor
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        # Constants for A&S approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
            BLOCK_SIZE=1024, num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # single scalar read; no torch tensor creation on host

        # Allocate threshold and output buffers in fp32
        thresh = torch.empty((total_rows, K), dtype=torch.float32, device=x.device)
        out_f32 = torch.empty_like(x_flat_f32)

        # Dynamic tiling for kernels
        BLOCK_SIZE_GATE, NUM_WARPS_GATE, NUM_STAGES_GATE = _choose_params(K)

        # 1) Compute per-row threshold and write across K
        apply_threshold_kernel[(total_rows,)](
            x_flat_f32, mean, std, z_buf, thresh,
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=NUM_WARPS_GATE, num_stages=NUM_STAGES_GATE
        )

        # 2) Apply gating (ReLU)
        gating_relu_kernel[(total_rows,)](
            x_flat_f32, thresh, out_f32,
            total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_GATE,
            num_warps=NUM_WARPS_GATE, num_stages=NUM_STAGES_GATE
        )

        # Reshape back and cast to bfloat16
        out = out_f32.view(*x.shape).to(torch.bfloat16)
        return out