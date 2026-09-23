import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row: pid indexes rows in [0, B*S)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    # Compute base linear index for this row in the flattened tensor
    base = b * S + s
    base_idx = base * D  # since each (b, s) row has D elements

    # Accumulators
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the feature dimension in tiles
    for off in range(0, D, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        # Load a tile; cast to float32 for accumulation
        x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    # Store per-row results
    tl.store(SUM_ptr + base, sum_val)
    tl.store(SUMSQ_ptr + base, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (z-score output)
):
    # Load scalar p
    p = tl.load(P_ptr)

    # Abramowitz & Stegun 5.2.23 approximation
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

    pl = tl.full((), p, tl.float32)
    mask_low = pl < p_low
    mask_mid = (pl >= p_low) & (pl <= p_high)
    mask_high = pl > p_high

    result = tl.zeros((), dtype=tl.float32)

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(pl))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    result_low = poly_low / den_low

    # Central region
    q_mid = pl - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    result_mid = poly_mid * q_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - pl))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    result_high = -poly_high / den_high

    # Select result based on region masks
    result = tl.where(mask_low, result_low, result)
    result = tl.where(mask_mid, result_mid, result)
    result = tl.where(mask_high, result_high, result)

    tl.store(OUT_ptr, result)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (b, s, tiles of D)
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguity
        assert inputs.is_cuda, "inputs must be CUDA tensors"
        inputs = inputs.contiguous()
        B, S, D = inputs.shape

        # Allocate device buffers for reductions and stats (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Device scalar for sparsity p
        p_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_buf.fill_(float(target_sparsity))

        # Buffer for z-score (device scalar)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Output buffer (float32 for kernel stores)
        out_f32 = torch.empty(B, S, D, dtype=torch.float32, device=inputs.device)

        # 1) Reduce sum and sumsq
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf,
            B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # 2) Compute mean and std
        compute_mean_std_kernel[grid_reduce](
            sum_buf, sumsq_buf, mean_buf, std_buf,
            D,
            num_warps=1,
        )

        # 3) Compute inverse normal CDF (z-score) for target_sparsity
        ndtri_approx_kernel[(1,)](  # single program
            p_buf, z_score_buf,
            num_warps=1,
        )

        # 4) Apply activation: y = max(0, x - (mean + std * z_score))
        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_score_buf, out_f32,
            B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # Cast to bfloat16 to match original function's return dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
