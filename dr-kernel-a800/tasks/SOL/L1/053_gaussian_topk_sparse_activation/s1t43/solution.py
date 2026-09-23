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
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Base offset for this (b, s) row (since tensor is contiguous with last dim D)
    base = (b * S + s) * D

    # Accumulators in float32
    total_sum = 0.0
    total_sumsq = 0.0

    # Loop over D in tiles
    for start in range(0, D, 1024):
        offs = start + tl.arange(0, 1024)
        mask = offs < D
        # Load a tile of the row
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        # Sum and sumsq for this tile
        tile_sum = tl.sum(x, axis=0)
        tile_sumsq = tl.sum(x * x, axis=0)
        total_sum += tile_sum
        total_sumsq += tile_sumsq

    # Store per-(b, s) sum and sumsq
    tl.store(SUM_ptr + pid, total_sum)
    tl.store(SUMSQ_ptr + pid, total_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    sum = tl.load(SUM_ptr + pid)
    sumsq = tl.load(SUMSQ_ptr + pid)

    # Compute mean and variance (population std, as in torch.std(unbiased=False))
    mean = sum / D
    var = sumsq / D - mean * mean
    # Clamp variance to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
    P,               # float32 scalar (host passes 1-element tensor via pointer)
):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF
    # We compute for the single element at OUT_ptr[0]
    p = tl.load(P)  # scalar float32

    # Constants for A&S approximation
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    y_low = -z_low

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    z_mid = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q2 / \
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    y_mid = z_mid

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1*q3 + c2)*q3 + c3)*q3 + c4)*q3 + c5)*q3 + c6) / \
             ((((d1*q3 + d2)*q3 + d3)*q3 + d4)*q3 + 1.0)
    y_high = z_high

    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)
    cond_high = p > p_high

    y = tl.where(cond_low, y_low, tl.where(cond_mid, y_mid, y_high))
    tl.store(OUT_ptr, y)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z_score)
    OUT_ptr,         # *float32, output buffer [B, S, D] contiguous
    B, S, D          # int32 dimensions
):
    # 2D grid: axis 0 over (b, s), axis 1 over tiles of D
    pid0 = tl.program_id(axis=0)  # 0..(B*S-1)
    pid1 = tl.program_id(axis=1)  # tile id along D

    b = pid0 // S
    s = pid0 % S

    # Load per-(b, s) mean and std
    mean = tl.load(MEAN_ptr + pid0)
    std = tl.load(STD_ptr + pid0)

    # Load scalar z_score
    z_score = tl.load(Z_ptr)  # scalar float32

    # Compute threshold (scalar) for this (b, s)
    threshold = mean + std * z_score

    # Base offset for this (b, s) row
    base = (b * S + s) * D

    # Tile offsets along D
    tile_start = pid1 * 1024
    offs = tile_start + tl.arange(0, 1024)
    mask = offs < D

    # Load input row tile
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)

    # Apply y = max(0, x - threshold)
    y = x - threshold
    y = tl.maximum(y, 0.0)

    # Store as float32; host will cast to bfloat16 if needed
    tl.store(OUT_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return inputs
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Ensure contiguous
        inputs = inputs.contiguous()
        B, S, D = inputs.shape

        # Prepare device buffers (float32 for computation)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Compute z_score via Triton (Abramowitz & Stegun 5.2.23)
        z_score = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p = torch.tensor(float(target_sparsity), dtype=torch.float32, device=inputs.device)
        ndtri_approx_kernel[(1,)](z_score, p)

        # Pass 1: reduce sum and sumsq
        reduce_sum_sumsq_kernel[(B * S,)](
            inputs, sum_buf, sumsq_buf, B, S, D,
            num_warps=4,
        )

        # Pass 2: compute mean and std
        compute_mean_std_kernel[(B * S,)](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D,
            num_warps=1,
        )

        # Output buffer in float32 for computation; cast to bfloat16 at the end
        out_f32 = torch.empty((B, S, D), dtype=torch.float32, device=inputs.device)

        # Pass 3: apply activation across full D for each (b, s)
        grid = (B * S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid](
            inputs, mean_buf, std_buf, z_score, out_f32, B, S, D,
            num_warps=8,
        )

        # Match original behavior: return bfloat16
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
