import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(X_ptr, OutSum_ptr, B: tl.int32, S: tl.int32, F: tl.int64,
                    stride_b: tl.int64, stride_s: tl.int64, stride_f: tl.int64,
                    BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per (b, s) row
    b = pid // S
    s = pid % S

    row_offset = b * stride_b + s * stride_s
    total = 0.0
    # Iterate over feature dimension in chunks
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
    # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / den_mid

    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    mask_low = p < p_low
    mask_up = p > (1.0 - p_low)
    z = z_mid
    z = tl.where(mask_low, z_low, z)
    z = tl.where(mask_up, z_up, z)

    tl.store(OutZ_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, Mean_ptr, Std_ptr, Out_ptr,
                           B: tl.int32, S: tl.int32, F: tl.int64,
                           stride_b: tl.int64, stride_s: tl.int64, stride_f: tl.int64,
                           BLOCK_F: tl.constexpr):
    pid = tl.program_id(axis=0)  # one program per (b, s) row
    b = pid // S
    s = pid % S
    mean = tl.load(Mean_ptr + pid)
    std = tl.load(Std_ptr + pid)
    threshold = mean + std  # scalar per row
    row_offset = b * stride_b + s * stride_s

    # Iterate across F in chunks, apply y = max(x - threshold, 0)
    for f_start in range(0, F, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        mask = f_offsets < F
        x_ptrs = X_ptr + row_offset + f_offsets * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        out_ptrs = Out_ptr + row_offset + f_offsets * stride_f
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function:
        - Compute per-row mean and std along feature dim (population std).
        - Compute inverse-normal CDF for target_sparsity (Abramowitz & Stegun 7.1.26).
        - Apply y = max(input - (mean + std * z), 0) and return bfloat16.
        """
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Choose block size for reduction along F
        BLOCK_F = 1024 if F >= 4096 else (512 if F >= 2048 else 256)
        num_warps = 4  # good default for these block sizes

        # Allocate accumulators
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels: one program per row
        grid = (B * S,)
        sum_rows_kernel[grid](
            inputs, out_sum,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=num_warps
        )
        sumsq_rows_kernel[grid](
            inputs, out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=num_warps
        )

        # Compute per-row mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](
            out_sum, out_sumsq, out_mean, out_std,
            B, S, F,
            num_warps=1
        )

        # Compute inverse-normal CDF for target_sparsity
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
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
            num_warps=1
        )
        z = z_buf[0]  # scalar float32

        # Apply threshold: y = max(x - (mean + std * z), 0)
        x_f32 = inputs.to(torch.float32)
        out_f32 = torch.empty_like(inputs, dtype=torch.float32)

        apply_threshold_kernel[grid](
            x_f32, out_mean, out_std, out_f32,
            B, S, F,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            BLOCK_F=BLOCK_F,
            num_warps=num_warps
        )

        # Cast to bfloat16 to match original return type
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
