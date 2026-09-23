import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_2d_kernel(x_ptr, mean_ptr, std_ptr, B, S, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (over last dim K) mean and std for each (b, s) row in x of shape [B, S, K].
    Output mean and std are stored as [B, S, 1].
    """
    row = tl.program_id(0)  # program id over total rows = B*S
    b = row // S
    s = row % S

    # Pointer to the start of this row
    row_base = (b * S + s) * K

    # Accumulators in fp32
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over K in tiles of BLOCK_SIZE
    # We iterate offset = 0..K with step BLOCK_SIZE, but each iteration runs a vectorized load
    for off in range(0, K, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        x_vals = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0)
        # Upcast to fp32 for stability
        x_vals = x_vals.to(tl.float32)
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    K_f = tl.float32(K)
    mean = sum_x / K_f
    # Population std (unbiased=False): sqrt(E[x^2] - mean^2)
    var = sum_x2 / K_f - mean * mean
    std = tl.sqrt(var)

    # Store mean and std to [B, S, 1]
    out_base = b * S + s  # index into [B, S] plane
    tl.store(mean_ptr + out_base, mean)
    tl.store(std_ptr + out_base, std)


@triton.jit
def ndtri_approx_kernel(z_buf, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5,
                         c1, c2, c3, c4, c5, c6,
                         d1, d2, d3, d4,
                         p_low, p_high, BLOCK_SIZE: tl.constexpr):
    """
    Compute z = inverse standard normal CDF for target_sparsity using A&S 5.2.23 approximation.
    Store result into z_buf[0] (single-element buffer).
    """
    t = target_sparsity  # scalar
    # Lower region
    q = tl.sqrt(-2.0 * tl.log(t))
    low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Upper region
    q2 = tl.sqrt(-2.0 * tl.log(1.0 - t))
    up = -(((((c1 * q2 + c2) * q2 + c3) * q2 + c4) * q2 + c5) * q2 + c6) / \
         ((((d1 * q2 + d2) * q2 + d3) * q2 + d4) * q2 + 1.0)

    # Central region
    q3 = t - 0.5
    r = q3 * q3
    c = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q3 / \
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Heuristic selection
    # If t < p_low: use low; if t > p_high: use up; else use c
    mask_low = t < p_low
    mask_high = t > p_high
    z = tl.where(mask_low, low, tl.where(mask_high, up, c))
    tl.store(z_buf, z)


@triton.jit
def broadcast_threshold_kernel(mean_ptr, std_ptr, z_ptr, threshold_ptr, B, S, K, BLOCK_SIZE: tl.constexpr):
    """
    For each (b, s), compute threshold = mean[b, s] + std[b, s] * z and broadcast across K into threshold[b, s, :].
    """
    row = tl.program_id(0)  # over B*S
    b = row // S
    s = row % S
    base = b * S + s

    # Load per-row mean and std
    mean = tl.load(mean_ptr + base)
    std = tl.load(std_ptr + base)
    z = tl.load(z_ptr)  # scalar
    thresh = mean + std * z

    # Write broadcast threshold across K elements
    for off in range(0, K, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        tl.store(threshold_ptr + (b * S + s) * K + idx, thresh, mask=mask)


@triton.jit
def gating_relu_2d_kernel(x_flat_ptr, threshold_flat_ptr, out_flat_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over flattened [total_rows, K] view:
    out[row, k] = max(x[row, k] - threshold[row, k], 0)
    """
    row = tl.program_id(0)  # over total_rows
    for off in range(0, K, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < K
        x = tl.load(x_flat_ptr + row * K + idx, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(threshold_flat_ptr + row * K + idx, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - t, 0.0)
        tl.store(out_flat_ptr + row * K + idx, y, mask=mask)


def _choose_params(K: int):
    # Heuristic tuning for Triton kernels
    if K >= 8192:
        return 4096, 8, 2
    else:
        return 2048, 4, 2


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        x: [B, S, K] input tensor (float16/float32/bfloat16), on CUDA.
        Returns sparsified output tensor [B, S, K] in bfloat16.
        """
        # Ensure CUDA tensors
        assert x.is_cuda, "Input must be a CUDA tensor for Triton kernels."
        B, S, K = x.shape
        total_rows = B * S

        # Compute mean and std in fp32 per row (B*S,1)
        x_f32 = x.to(torch.float32)
        mean = torch.empty((B, S, 1), dtype=torch.float32, device=x.device)
        std = torch.empty((B, S, 1), dtype=torch.float32, device=x.device)

        BLOCK_STATS = 1024
        grid_stats = (total_rows,)
        compute_row_stats_2d_kernel[grid_stats](
            x_f32.view(-1), mean.view(-1), std.view(-1),
            B, S, K,
            BLOCK_SIZE=BLOCK_STATS,
            num_warps=4, num_stages=2
        )

        # Compute z = _ndtri(target_sparsity) in Triton (no torch.tensor/distributions)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        ndtri_approx_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            BLOCK_SIZE=1024,
            num_warps=1, num_stages=1
        )
        z = float(z_buf.item())  # single scalar read; no torch tensor creation on host

        # Compute per-row threshold broadcasted to [B, S, K] in fp32
        threshold = torch.empty((B, S, K), dtype=torch.float32, device=x.device)
        BLOCK_BCAST = 2048 if K < 8192 else 4096
        grid_bcast = (total_rows,)
        broadcast_threshold_kernel[grid_bcast](
            mean.view(-1), std.view(-1), z_buf, threshold.view(-1),
            B, S, K,
            BLOCK_SIZE=BLOCK_BCAST,
            num_warps=4, num_stages=2
        )

        # Flatten views for elementwise gating
        x_flat = x_f32.view(-1)
        thr_flat = threshold.view(-1)
        out_flat = torch.empty_like(x_flat)

        BLOCK_GATE, NUM_WARPS_GATE, NUM_STAGES_GATE = _choose_params(K)
        grid_gate = (total_rows,)
        gating_relu_2d_kernel[grid_gate](
            x_flat, thr_flat, out_flat,
            total_rows, K,
            BLOCK_SIZE=BLOCK_GATE,
            num_warps=NUM_WARPS_GATE, num_stages=NUM_STAGES_GATE
        )

        # Reshape back and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, K).to(torch.bfloat16)
        return out