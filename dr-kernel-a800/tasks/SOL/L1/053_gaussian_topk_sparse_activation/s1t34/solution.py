import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 scalars
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # For contiguous [B, S, D], linear index for row (b, s, :) is base = (b*S + s) * D
    base = (b * S + s) * D

    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, total_sum)
    tl.store(SUMSQ_ptr + pid, total_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D,         # int32 scalars
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def compute_zscore_kernel(
    P_ptr,           # *float32, length 1 (scalar sparsity p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p (assumed to be in [0, 1])
    p = tl.load(P_ptr)
    # Constants for A&S 5.2.23 approximation (Abramowitz & Stegun)
    # Note: for p > 0.5, use symmetry: ndtri(p) = -ndtri(1-p)
    if p > 0.5:
        p = 1.0 - p
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

    # Lower region
    p_low = 0.02425
    p_low_ = p_low
    mask_low = p < p_low_
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    qf_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = poly_low / qf_low

    # Central region
    p_high = 1.0 - p_low_
    mask_mid = (p >= p_low_) & (p <= p_high)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    qf_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / qf_mid

    # Upper region
    mask_high = (p > p_high)
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
    qf_up = (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
    z_up = -poly_up / qf_up

    # Select the appropriate branch (p was adjusted to <= 0.5)
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_up, 0.0)

    # Store z score
    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bf16, output [B, S, D]
    B, S, D,         # int32 scalars
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # linear index for this row

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    y_bf16 = y.to(tl.bfloat16)
    tl.store(OUT_ptr + base_idx + offs, y_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float = 0.0):
        # Ensure input is on CUDA and contiguous (no torch operations on tensors)
        if not inputs.is_cuda:
            inputs = inputs.to('cuda')
        inputs = inputs.contiguous()
        device = inputs.device
        B = inputs.shape[0]
        S = inputs.shape[1]
        D = inputs.shape[2]

        # Allocate device buffers (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=device)

        # 1) Reduce to per-row sum and sumsq
        grid_reduce = (B * S,)
        reduce_mean_std_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=2048, num_warps=8
        )

        # 2) Compute mean and std per row
        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D
        )

        # 3) Compute z-score via Triton kernel (inverse normal CDF)
        # Pass sparsity as a device scalar tensor on the correct device
        p_dev = torch.tensor(float(target_sparsity), dtype=torch.float32, device=device)
        compute_zscore_kernel[(1,)](p_dev, z_score_buf, num_warps=1)

        # 4) Apply activation: y = max(0, x - (mean + std * z_score)), store as bf16
        out = torch.empty((B, S, D), dtype=torch.bfloat16, device=device)
        grid_apply = (B, S, triton.cdiv(D, 512))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_score_buf, out, B, S, D, BLOCK_SIZE=512, num_warps=8
        )

        return out


def run(*args):
    return ModelNew()(*args)
