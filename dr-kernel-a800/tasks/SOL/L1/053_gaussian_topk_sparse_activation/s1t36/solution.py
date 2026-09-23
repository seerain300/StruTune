import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S
    d = b * S + s

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over feature dimension D in tiles
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        base = d * D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32 number of features
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
    OUT_ptr,         # *float32, length 1 (output z-score)
    p,               # scalar float32 (target sparsity in [0,1])
):
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region constants
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
    q_low = tl.sqrt(-2.0 * tl.log(p))
    res_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
              ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    z_low = -res_low

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    res_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
               ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    z_high = res_high

    # Select region
    z = tl.where(p < p_low, z_low, tl.where(p > p_high, z_high, z_mid))

    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bf16 output [B, S, D]
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    d = b * S + s
    base = d * D  # scalar offset into contiguous [B, S, D]

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + d)  # scalar float32
    std = tl.load(STD_ptr + d)    # scalar float32
    z_score = tl.load(Z_ptr)      # scalar float32

    threshold = mean + std * z_score

    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    # Store as bfloat16 to match original behavior
    y_bf16 = y.to(tl.bfloat16)
    tl.store(OUT_ptr + base + offs, y_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return input unchanged but cast to bfloat16
        if target_sparsity == 0.0:
            return inputs.to(torch.bfloat16)

        # Expect 3D input: [batch_size, seq_len, intermediate_size]
        assert inputs.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, D = inputs.shape
        device = inputs.device

        # Make input contiguous
        X = inputs.contiguous()

        # Output in bfloat16 to match original behavior
        OUT = torch.empty((B, S, D), dtype=torch.bfloat16, device=device)

        # Intermediate buffers as float32
        sum_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        sumsq_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        mean_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        std_buf = torch.empty((B * S,), dtype=torch.float32, device=device)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # 1) Reduce sum and sumsq (no host torch ops)
        BLOCK_SIZE = 1024
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            X, sum_buf, sumsq_buf,
            B, S, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        # 2) Compute mean and std (no host torch ops)
        grid_mean = (B * S,)
        compute_mean_std_kernel[grid_mean](
            sum_buf, sumsq_buf, mean_buf, std_buf,
            D,
            num_warps=1,
        )

        # 3) Compute ndtri(z) for target sparsity in Triton (no host torch ops)
        # p is a scalar float32; OUT buffer has 1 element for z_score
        ndtri_approx_kernel[(1,)](
            z_buf,
            float(target_sparsity),
            num_warps=1,
        )

        # 4) Apply activation: y = max(0, x - (mean + std * z)) (no host torch ops)
        BLOCK_SIZE_ACT = 1024
        grid_apply = (B, S, triton.cdiv(D, BLOCK_SIZE_ACT))
        apply_activation_kernel[grid_apply](
            X, mean_buf, std_buf, z_buf, OUT,
            B, S, D,
            BLOCK_SIZE=BLOCK_SIZE_ACT,
            num_warps=8,
        )

        return OUT


def run(*args):
    return ModelNew()(*args)
