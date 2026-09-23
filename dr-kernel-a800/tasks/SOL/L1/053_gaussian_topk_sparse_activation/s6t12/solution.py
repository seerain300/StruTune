import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(X_ptr, OutSum_ptr, B: tl.int32, S: tl.int32, F: tl.int32,
                    stride_b: tl.int64, stride_s: tl.int64, stride_f: tl.int64,
                    BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_offset = b * stride_b + s * stride_s
    total = 0.0
    for f_start in range(0, F, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        mask = f_offsets < F
        x_ptrs = X_ptr + row_offset + f_offsets * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
    tl.store(OutSum_ptr + pid, total)


@triton.jit
def sumsq_rows_kernel(X_ptr, OutSumSq_ptr, B: tl.int32, S: tl.int32, F: tl.int64,
                      stride_b: tl.int64, stride_s: tl.int64, stride_f: tl.int64,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    row_offset = b * stride_b + s * stride_s
    total = 0.0
    for f_start in range(0, F, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        mask = f_offsets < F
        x_ptrs = X_ptr + row_offset + f_offsets * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.store(OutSumSq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(OutSum_ptr, OutSumSq_ptr, OutMean_ptr, OutStd_ptr,
                         B: tl.int32, S: tl.int32, F: tl.int64):
    pid = tl.program_id(axis=0)  # one program per row
    sum_val = tl.load(OutSum_ptr + pid)
    sumsq_val = tl.load(OutSumSq_ptr + pid)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(OutMean_ptr + pid, mean)
    tl.store(OutStd_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(OutZ_ptr, p: tl.float32,
                        a1: tl.float32, a2: tl.float32, a3: tl.float32, a4: tl.float32, a5: tl.float32, a6: tl.float32,
                        b1: tl.float32, b2: tl.float32, b3: tl.float32, b4: tl.float32, b5: tl.float32,
                        c1: tl.float32, c2: tl.float32, c3: tl.float32, c4: tl.float32, c5: tl.float32, c6: tl.float32,
                        d1: tl.float32, d2: tl.float32, d3: tl.float32, d4: tl.float32,
                        p_low: tl.float32):
    # Abramowitz & Stegun 7.1.26 approximation (inverse normal CDF)
    # z = sign(p - 0.5) * ((a1 t + a2) t + a3) t + a4) t + a5) t + a6) / (((b1 t + b2) t + b3) t + b4) t + 1.0)
    # for t = sqrt(2) * |p - 0.5|
    # with adjustments for p < p_low and p > (1 - p_low).
    sqrt2 = 1.4142135623730951
    one = 1.0

    # Masks
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_upper = p > (1.0 - p_low)
    mask_lower = p < p_low

    # Mid region
    t_mid = sqrt2 * (p - 0.5)
    z_mid = (((((a1 * t_mid + a2) * t_mid + a3) * t_mid + a4) * t_mid + a5) * t_mid + a6) / \
            (((((b1 * t_mid + b2) * t_mid + b3) * t_mid + b4) * t_mid) + one)
    z_mid = tl.where(t_mid < 0.0, -z_mid, z_mid)  # sign adjustment if needed

    # Lower region
    q_lower = tl.sqrt(-2.0 * tl.log(p))
    z_lower = (((((c1 * q_lower + c2) * q_lower + c3) * q_lower + c4) * q_lower + c5) * q_lower + c6) / \
              (((((d1 * q_lower + d2) * q_lower + d3) * q_lower + d4) * q_lower + one))

    # Upper region
    q_upper = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_upper = -(((((c1 * q_upper + c2) * q_upper + c3) * q_upper + c4) * q_upper + c5) * q_upper + c6) / \
              (((((d1 * q_upper + d2) * q_upper + d3) * q_upper + d4) * q_upper + one))

    # Select by mask
    z = tl.where(mask_mid, z_mid, 0.0)
    z = tl.where(mask_upper, z_upper, z)
    z = tl.where(mask_lower, z_lower, z)

    tl.store(OutZ_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, Out_ptr, Mean_ptr, Std_ptr, Z_ptr,
                           B: tl.int32, S: tl.int32, F: tl.int64,
                           stride_b: tl.int64, stride_s: tl.int64, stride_f: tl.int64):
    pid = tl.program_id(axis=0)  # one program per row
    b = pid // S
    s = pid % S

    row_offset = b * stride_b + s * stride_s
    mean = tl.load(Mean_ptr + pid)
    std = tl.load(Std_ptr + pid)
    z = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z

    for f_start in range(0, F, 1024):
        f_offsets = f_start + tl.arange(0, 1024)
        mask = f_offsets < F
        x_ptrs = X_ptr + row_offset + f_offsets * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        out_ptrs = Out_ptr + row_offset + f_offsets * stride_f
        tl.store(out_ptrs, y, mask=mask)


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-based implementation of run:
    - Compute per-row (b, s) mean and std along feature dim F.
    - Compute z = inverse-normal CDF at target_sparsity (Abramowitz & Stegun 7.1.26).
    - Apply threshold: y = max(input - (mean + std * z), 0).
    - Return bfloat16.
    """
    assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
    inputs = inputs.contiguous()
    B, S, F = inputs.shape
    device = inputs.device

    # Prepare outputs for reductions
    out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
    out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
    out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
    out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

    # Choose block size for feature reduction
    BLOCK_F = 1024 if F >= 2048 else 512
    grid = (B * S,)

    # Compute sums and sumsq
    sum_rows_kernel[grid](inputs, out_sum, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=8)
    sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2), BLOCK_F=BLOCK_F, num_warps=8)

    # Compute mean and std per row
    compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F, num_warps=4)

    # Compute inverse-normal CDF for scalar target_sparsity
    z_buf = torch.empty((1,), dtype=torch.float32, device=device)
    # Constants from Abramowitz & Stegun 7.1.26
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
    p_low = 0.02425
    ndtri_scalar_kernel[(1,)](z_buf, float(target_sparsity),
                             a1, a2, a3, a4, a5, a6,
                             b1, b2, b3, b4, b5,
                             c1, c2, c3, c4, c5, c6,
                             d1, d2, d3, d4,
                             p_low, num_warps=1)

    # Prepare output buffer as float32
    out_f32 = torch.empty_like(inputs, dtype=torch.float32)

    # Apply threshold per row
    apply_threshold_kernel[(B * S,)](
        inputs, out_f32, out_mean, out_std, z_buf,
        B, S, F, inputs.stride(0), inputs.stride(1), inputs.stride(2),
        num_warps=4
    )

    # Cast to bfloat16 to match original return type
    return out_f32.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape [batch_size, seq_len, intermediate_size]
        inputs = args[0]
        target_sparsity = 0.05  # default; adjust if needed
        return run(inputs, target_sparsity)


def run(*args):
    return ModelNew()(*args)
