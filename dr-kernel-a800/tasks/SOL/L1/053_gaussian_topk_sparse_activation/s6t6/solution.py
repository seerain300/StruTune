import torch
import triton
import triton.language as tl


# Kernel: compute sum across the feature dimension for each row (b, s)
@triton.jit
def sum_rows_kernel(
    inp_ptr,               # *const float32 (we cast input in-kernel)
    out_sum_ptr,           # *float32, shape [B*S]
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*S-1
    b = pid // S
    s = pid % S
    row_base = b * stride_b + s * stride_s

    acc = 0.0
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        # Load vector from the row: address = row_base + offs * stride_f
        ptrs = inp_ptr + row_base + offs * stride_f
        # The input tensor may be bfloat16; cast to float32 for accumulation
        vec = tl.load(ptrs, mask=mask, other=0.0)
        vec_f32 = vec.to(tl.float32)
        # Reduce this chunk to scalar and accumulate
        acc += tl.sum(vec_f32, axis=0)

    # Write per-row sum
    tl.atomic_add(out_sum_ptr + pid, acc)


# Kernel: compute sum of squares across the feature dimension for each row
@triton.jit
def sumsq_rows_kernel(
    inp_ptr,               # *const float32 (we cast input in-kernel)
    out_sumsq_ptr,         # *float32, shape [B*S]
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*S-1
    b = pid // S
    s = pid % S
    row_base = b * stride_b + s * stride_s

    acc = 0.0
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs = inp_ptr + row_base + offs * stride_f
        vec = tl.load(ptrs, mask=mask, other=0.0)
        vec_f32 = vec.to(tl.float32)
        acc += tl.sum(vec_f32 * vec_f32, axis=0)

    tl.atomic_add(out_sumsq_ptr + pid, acc)


# Kernel: compute mean and std from sums and sumsq (population std)
@triton.jit
def compute_stats_kernel(
    out_sum_ptr,           # *const float32, shape [B*S]
    out_sumsq_ptr,         # *const float32, shape [B*S]
    out_mean_ptr,          # *float32, shape [B*S]
    out_std_ptr,           # *float32, shape [B*S]
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*S-1
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    # mean and population variance
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    # ensure non-negative due to numerical reasons
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


# Kernel: compute inverse-normal CDF (Abramowitz & Stegun 7.1.26) for a single p (scalar)
@triton.jit
def ndtri_scalar_kernel(
    z_out_ptr,             # *float32, length 1
    p,                     # scalar float32 (target_sparsity)
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,                 # scalar float32
):
    # Vectorize over a single element using mask
    mask = tl.full((1,), True, tl.bool_)
    # Lower region
    # q = sqrt(-2 * log(p)) for p in (0, p_low)
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # Mid region: central part
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    # Upper region: p >= 1 - p_low
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select based on p using masks (no Python branching)
    sel_low = (p > 0.0) & (p < p_low)
    sel_up = (p > (1.0 - p_low))
    # Default mid
    z = z_mid
    # If in low region, use z_low; if in up region, use z_up
    z = tl.where(sel_low, z_low, z)
    z = tl.where(sel_up, z_up, z)

    # Store to z_out (only one element)
    tl.store(z_out_ptr, z, mask=mask)


# Kernel: apply threshold and ReLU: y = max(inp - (mean + std * z), 0)
@triton.jit
def apply_threshold_kernel(
    inp_ptr,               # *const float32 (we cast input in-kernel)
    mean_ptr,              # *const float32, shape [B*S]
    std_ptr,               # *const float32, shape [B*S]
    z_ptr,                 # *const float32, length 1
    out_ptr,               # *float32, shape [B*S*F] (we store elementwise)
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..B*S-1
    b = pid // S
    s = pid % S
    row_base = b * stride_b + s * stride_s

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar

    threshold = mean + std * z

    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs_in = inp_ptr + row_base + offs * stride_f
        inp_vec = tl.load(ptrs_in, mask=mask, other=0.0)
        inp_f32 = inp_vec.to(tl.float32)
        # y = max(inp - threshold, 0)
        y_vec = tl.maximum(inp_f32 - threshold, 0.0)
        # Store to output (float32)
        ptrs_out = out_ptr + pid * F + start + tl.arange(0, BLOCK_F)
        tl.store(ptrs_out, y_vec, mask=mask)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized Gaussian-based top-k sparse activation:
    - Compute per-(batch, seq) mean and std along feature dim (population std).
    - Compute inverse-normal CDF for target_sparsity via A&S 7.1.26 approximation in Triton.
    - Apply y = max(input - (mean + std * z), 0), return bfloat16.
    """
    assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
    # Ensure contiguity along feature dim
    inputs = inputs.contiguous()
    B, S, F = inputs.shape
    device = inputs.device
    dtype = inputs.dtype  # may be bfloat16; we cast inside kernels

    # Prepare pointers; we will cast to float32 inside Triton kernels for accumulation and elementwise ops
    # Allocate per-row sums
    out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
    out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

    # Launch sum and sumsq kernels
    BLOCK_F = min(1024, _next_pow2(F))
    grid = (B * S,)
    sum_rows_kernel[grid](
        inputs, out_sum,
        B, S, F,
        inputs.stride(0), inputs.stride(1), inputs.stride(2),
        BLOCK_F=BLOCK_F,
    )
    sumsq_rows_kernel[grid](
        inputs, out_sumsq,
        B, S, F,
        inputs.stride(0), inputs.stride(1), inputs.stride(2),
        BLOCK_F=BLOCK_F,
    )

    # Compute mean and std (population)
    out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
    out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
    compute_stats_kernel[grid](
        out_sum, out_sumsq,
        out_mean, out_std,
        B, S, F,
    )

    # Compute inverse-normal CDF for scalar target_sparsity in Triton
    z_buf = torch.empty((1,), dtype=torch.float32, device=device)  # device scalar
    # Constants for A&S 7.1.26 approximation
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
    p_low = 0.02425

    ndtri_scalar_kernel[(1,)](
        z_buf, float(target_sparsity),
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5,
        c1, c2, c3, c4, c5, c6,
        d1, d2, d3, d4,
        p_low,
    )

    # Prepare output as float32 (we'll cast to bfloat16 after Triton)
    out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

    # Apply threshold and ReLU
    apply_threshold_kernel[grid](
        inputs,              # cast inside kernel
        out_mean, out_std,
        z_buf,
        out_f32,
        B, S, F,
        inputs.stride(0), inputs.stride(1), inputs.stride(2),
        BLOCK_F=BLOCK_F,
    )

    # Cast to bfloat16 to match original model's return type
    return out_f32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input: [B, S, F]
        assert len(args) == 1, "ModelNew.forward expects a single tensor input"
        inputs = args[0]
        return run(inputs, 0.02)  # target_sparsity is fixed as per original code


def run(*args):
    return ModelNew()(*args)
