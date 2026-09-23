import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base_idx = b * S * D + s * D  # scalar offset for this row
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # Iterate over D in chunks
    num_chunks = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        offs = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
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
    P_ptr,           # *float32, length 1 (scalar sparsity p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load p (inverse normal CDF for standard normal)
    p = tl.load(P_ptr)
    # Constants for Abramowitz & Stegun 5.2.23 approximation
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

    # Compute z for each region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / den_mid

    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))

    # Select result based on mask
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous (float32)
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # scalar int32 base offset

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load mean and std for this (b, s) row
    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar z-score
    threshold = mean + std * z_score

    # Load input, apply activation
    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold  # broadcast scalar threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure CUDA device and contiguity
        if not inputs.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors.")
        inputs = inputs.contiguous()
        B, S, D = inputs.shape

        # Buffers for sums, mean, std (float32 on device)
        sums = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        means = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        stds = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Kernel 1: reduce to sum and sumsq (float32 input, output float32 buffers)
        grid1 = (B * S,)
        BLOCK_SIZE1 = 1024
        triton.run(reduce_mean_std_kernel, grid1, inputs, sums, sumsq, B, S, D, BLOCK_SIZE=BLOCK_SIZE1, num_warps=8)

        # Kernel 2: compute mean and std
        grid2 = (B * S,)
        triton.run(compute_mean_std_kernel, grid2, sums, sumsq, means, stds, D, num_warps=4)

        # Kernel 3: compute inverse normal CDF (z-score) for target_sparsity
        p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_tensor[0] = float(target_sparsity)  # initialize scalar on device
        z_score = torch.empty(1, dtype=torch.float32, device=inputs.device)
        triton.run(ndtri_approx_kernel, (1,), p_tensor, z_score, num_warps=1)

        # Kernel 4: apply activation (x - threshold) with ReLU, store as float32
        x_f32 = inputs.to(torch.float32)
        out_f32 = torch.empty_like(x_f32)
        grid3 = (B, S, triton.cdiv(D, 1024))
        triton.run(apply_activation_kernel, grid3, x_f32, means, stds, z_score, out_f32, B, S, D, BLOCK_SIZE=1024, num_warps=8)

        # Cast to bfloat16 to match original function’s output dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
