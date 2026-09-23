import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    # Map program id to (b, s)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s) row
    base = (b * S + s) * D  # since contiguous, index = (b*S + s)*D + d

    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over feature dimension in tiles of BLOCK_SIZE
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        # Load a tile of the row, masked
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
        # Accumulate in float32
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    # Store per-row sums
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sum_sq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    # Load sums
    sum_val = tl.load(SUM_ptr + pid)
    sum_sq = tl.load(SUMSQ_ptr + pid)

    # Compute mean and variance
    mean = sum_val / D
    var = sum_sq / D - mean * mean
    # Guard against tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (input scalar p)
    Z_ptr,           # *float32, length 1 (output scalar z = ndtri(p))
    # Constants for Abramowitz & Stegun 5.2.23 approximation
):
    p = tl.load(P_ptr)
    # Lower region
    p_low = 0.02425
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
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
        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00
        c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00

        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        denom = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = poly / denom
    else:
        # Central region
        # For p close to 0.5, we use the central approximation; however, since
        # p_low is > 0.5 in this task, we can use a simpler central formula.
        # We'll implement the central formula with a general p in [p_low, ~0.975].
        q = p - 0.5
        r = q * q
        a = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        b = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        z = (a * q) / b

    # Upper region
    p_high = 1.0 - p_low
    if p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        denom = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = -poly / denom

    tl.store(Z_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous, original dtype
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1, z_score scalar
    OUT_ptr,         # *float32, length B*S*D (intermediate output)
    B, S, D          # int32 dimensions
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    tile = tl.program_id(2)

    # Compute offsets for this (b, s) and tile along D
    D_tiles = tl.cdiv(D, BLOCK_SIZE)
    if tile >= D_tiles:
        return

    d_start = tile * BLOCK_SIZE
    idx = d_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < D

    # Row base index
    row_base = (b * S + s) * D

    # Load x tile
    x = tl.load(X_ptr + row_base + idx, mask=mask, other=0.0).to(tl.float32)

    # Load mean and std for this row
    mean = tl.load(MEAN_ptr + (b * S + s))
    std = tl.load(STD_ptr + (b * S + s))
    z_score = tl.load(Z_ptr)  # scalar

    # Compute threshold and apply activation
    threshold = mean + std * z_score
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store intermediate result (float32). Note: OUT is float32 buffer.
    tl.store(OUT_ptr + (b * S + s) * D + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure device and contiguity
        assert inputs.is_cuda, "Inputs must be on CUDA for Triton kernels"
        inputs = inputs.contiguous()

        B, S, D = inputs.shape
        # Allocate device buffers for sums, sumsq, mean, std, z_score, and output
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s) row
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=1024, num_warps=8
        )

        # Launch mean/std kernel: one program per (b, s)
        compute_mean_std_kernel[(B * S,)](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D, num_warps=1
        )

        # Compute z_score via Triton approximation (single program)
        target = torch.empty(1, dtype=torch.float32, device=inputs.device)
        target.fill_(float(target_sparsity))
        ndtri_approx_kernel[(1,)](
            target, z_score_buf,  # pass empty P to allocate? No, we filled target. We need to pass target.
            # Note: we filled 'target' with the desired sparsity p. The kernel uses P_ptr = target.
        )

        # Apply activation: 3D grid over (b, s, tiles of D)
        out_f32 = torch.empty(B * S * D, dtype=torch.float32, device=inputs.device)
        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_score_buf, out_f32, B, S, D, BLOCK_SIZE=1024, num_warps=8
        )

        # Return bfloat16 to match original behavior
        return out_f32.view(B, S, D).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
