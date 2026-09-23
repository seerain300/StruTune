import torch
import triton
import triton.language as tl


@triton.jit
def single_reduce_rows_kernel(
    x_ptr,                # *const float32
    out_sum_ptr,          # *float32, shape [B*S]
    out_sumsq_ptr,        # *float32, shape [B*S]
    B: tl.constexpr,      # number of batches
    S: tl.constexpr,      # number of seq positions
    F: tl.constexpr,      # feature dimension
    stride_b,             # x.stride(0)
    stride_s,             # x.stride(1)
    stride_f,             # x.stride(2)
    BLOCK_F: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(0)
    # Compute b, s from linear pid
    b = pid // S
    s = pid % S

    # Pointer to the start of this row
    row_ptr = x_ptr + b * stride_b + s * stride_s

    # Accumulators for this row
    acc_sum = 0.0
    acc_sumsq = 0.0

    # Loop over feature dimension in chunks
    for offset in range(0, F, BLOCK_F):
        cols = offset + tl.arange(0, BLOCK_F)
        mask = cols < F
        ptrs = row_ptr + cols * stride_f
        # Load as float32
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # Reduce this chunk
        chunk_sum = tl.sum(vals, axis=0)
        chunk_sumsq = tl.sum(vals * vals, axis=0)
        acc_sum += chunk_sum
        acc_sumsq += chunk_sumsq

    # Store per-row results
    tl.atomic_add(out_sum_ptr + pid, acc_sum)
    tl.atomic_add(out_sumsq_ptr + pid, acc_sumsq)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,   # *const float32, shape [B*S]
    out_sumsq_ptr, # *const float32, shape [B*S]
    out_mean_ptr,  # *float32, shape [B*S]
    out_std_ptr,   # *float32, shape [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    # mean = sum / F
    mean = sum_val / F
    # var = E[x^2] - mean^2
    var = sumsq_val / F - mean * mean
    # clamp to non-negative
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    z_out_ptr,          # *float32, shape [1]
    p,                  # float32 scalar target_sparsity
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,
):
    # Single-program kernel computing inverse-normal CDF
    # Abramowitz & Stegun 7.1.26 approximation
    # q = sqrt(2) * (p_low <= p < 1 - p_low)
    sqrt2 = 1.4142135623730951
    q = sqrt2 * p  # z ~ sqrt(2) * (1 - p) for lower; use p for upper

    # Region masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    mask_high = p > (1.0 - p_low)

    # Compute t for each region using masks
    # Lower region: t = (1 - p)/p
    t_low = (1.0 - p) / p
    # Mid region: t = 0 (we'll use polynomial)
    t_mid = 0.0
    # High region: t = (p - 1)/p
    t_high = (p - 1.0) / p

    # Polynomial for w and z per region
    # w = 1 / (1 + t * (c1 + t*(c2 + t*(c3 + t*(c4 + t*c5)))))
    # z = (((((a1*t + a2)*t + a3)*t + a4)*t + a5)*t + a6) * q / w
    # Combine using masks
    c_poly = c1 + t_mid * (c2 + t_mid * (c3 + t_mid * (c4 + t_mid * c5)))
    w_mid = 1.0 / (1.0 + t_mid * (c1 + t_mid * (c2 + t_mid * (c3 + t_mid * (c4 + t_mid * c5)))))
    num_mid = (((((a1 * t_mid + a2) * t_mid + a3) * t_mid + a4) * t_mid + a5) * t_mid + a6) * q
    z_mid = num_mid / w_mid

    a_poly = a1 + t_low * (a2 + t_low * (a3 + t_low * (a4 + t_low * (a5 + t_low * a6))))
    w_low = 1.0 / (1.0 + t_low * (c1 + t_low * (c2 + t_low * (c3 + t_low * (c4 + t_low * c5)))))
    num_low = (((((a1 * t_low + a2) * t_low + a3) * t_low + a4) * t_low + a5) * t_low + a6) * q
    z_low = num_low / w_low

    b_poly = b1 + t_high * (b2 + t_high * (b3 + t_high * (b4 + t_high * b5)))
    w_high = 1.0 / (1.0 + t_high * (c1 + t_high * (c2 + t_high * (c3 + t_high * (c4 + t_high * c5)))))
    num_high = (((((a1 * t_high + a2) * t_high + a3) * t_high + a4) * t_high + a5) * t_high + a6) * q
    z_high = - (num_high / w_high)  # upper region is negative

    # Combine via masks
    z_val = z_mid + (z_low - z_mid) * mask_low + (z_high - z_mid) * mask_high

    # Store to output
    tl.store(z_out_ptr, z_val)


@triton.jit
def apply_threshold_kernel(
    x_ptr,               # *const float32, input
    mean_ptr,            # *const float32, shape [B*S]
    std_ptr,             # *const float32, shape [B*S]
    z_ptr,               # *const float32, shape [1]
    out_ptr,             # *float32, output (we'll cast to bfloat16 in host)
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    stride_b,
    stride_s,
    stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar z
    threshold = mean + std * z

    row_ptr = x_ptr + b * stride_b + s * stride_s
    out_row_ptr = out_ptr + b * stride_b + s * stride_s

    for offset in range(0, F, BLOCK_F):
        cols = offset + tl.arange(0, BLOCK_F)
        mask = cols < F
        x_vals = tl.load(row_ptr + cols * stride_f, mask=mask, other=0.0)
        y_vals = tl.maximum(x_vals - threshold, 0.0)
        tl.store(out_row_ptr + cols * stride_f, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function:
        - Computes per-(batch, seq) mean and std along the feature dimension.
        - Computes inverse-normal CDF for target_sparsity in Triton.
        - Applies y = max(input - (mean + std * z), 0), returns bfloat16.
        """
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Output in float32 for stable computation, cast to bfloat16 at the end
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        # Choose block size for feature reduction; 1024 works well for large F
        BLOCK_F = 1024 if F >= 1024 else (512 if F >= 512 else 256)
        grid = (B * S,)

        # Single pass to compute sum and sumsq
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
        single_reduce_rows_kernel[grid](
            inputs, out_sum, out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # Compute per-row mean and std
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F)

        # Inverse-normal CDF for target_sparsity (Abramowitz & Stegun 7.1.26) in Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        ndtri_scalar_kernel[(1,)](
            z_buf,
            float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
        )

        # Apply threshold and ReLU in Triton
        apply_threshold_kernel[grid](
            inputs, out_mean, out_std, z_buf, out_f32,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # Cast to bfloat16 to match original output dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
