import torch
import triton
import triton.language as tl


# -----------------------
# Triton kernels
# -----------------------

@triton.jit
def sum_rows_kernel(X_ptr, out_sum_ptr,
                     B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                     stride_b, stride_s, stride_f,
                     BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_start = b * stride_b + s * stride_s

    total_sum = 0.0
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)

    tl.atomic_add(out_sum_ptr + pid, total_sum)


@triton.jit
def sumsq_rows_kernel(X_ptr, out_sumsq_ptr,
                      B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                      stride_b, stride_s, stride_f,
                      BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_start = b * stride_b + s * stride_s

    total_sumsq = 0.0
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        total_sumsq += tl.sum(x * x, axis=0)

    tl.atomic_add(out_sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_stats_kernel(out_sum_ptr, out_sumsq_ptr, out_mean_ptr, out_std_ptr,
                         B: tl.constexpr, S: tl.constexpr, F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)

    F_f = tl.float32(F)
    mean = sum_val / F_f
    var = sumsq_val / F_f - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(out_ptr, p, a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4, p_low):
    # Compute inverse normal CDF at p using A&S 7.1.26 approximation.
    p = tl.float32(p)  # ensure fp32 scalar

    # lower region: p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # mid region: p_low <= p <= 1 - p_low
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    poly_denom = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / poly_denom

    # upper region: p > 1 - p_low
    p_high = 1.0 - p_low
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    mask_low = p < p_low
    mask_high = p > p_high

    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_high, z_high, z)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, mean_ptr, std_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b_x, stride_s_x, stride_f_x,
                           stride_b_out, stride_s_out, stride_f_out,
                           BLOCK_F: tl.constexpr):
    # Each program handles one row (b, s) and processes features in chunks of BLOCK_F.
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_mean = tl.load(mean_ptr + pid)
    row_std = tl.load(std_ptr + pid)

    threshold = row_mean + row_std * tl.load(out_ptr)  # out_ptr points to 1-element tensor with z

    row_start_x = b * stride_b_x + s * stride_s_x
    row_start_out = b * stride_b_out + s * stride_s_out

    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        x_ptrs = X_ptr + row_start_x + cols * stride_f_x
        y_ptrs = out_ptr + row_start_out + cols * stride_f_out  # use out_ptr as output (float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        tl.store(y_ptrs, y, mask=mask)


# -----------------------
# Host-side helpers
# -----------------------

def _pick_block_f(F: int) -> int:
    if F >= 2048:
        return 1024
    elif F >= 1024:
        return 512
    else:
        return 256


# -----------------------
# ModelNew
# -----------------------

class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only forward: no torch ops in here.
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # 1) Compute per-row sum and sumsq
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        BLOCK_F = _pick_block_f(F)
        grid = (B * S,)
        sum_rows_kernel[grid](
            inputs, out_sum,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F
        )
        sumsq_rows_kernel[grid](
            inputs, out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            BLOCK_F=BLOCK_F
        )

        # 2) Compute mean and std (population)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F)

        # 3) Compute inverse-normal CDF for scalar target_sparsity (A&S 7.1.26) in Triton
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        ndtri_scalar_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4, p_low
        )

        z = z_buf[0]  # scalar

        # 4) Apply threshold per row
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        grid_apply = (B * S,)
        apply_threshold_kernel[grid_apply](
            inputs, out_mean, out_std, out_f32,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            out_f32.stride(0), out_f32.stride(1), out_f32.stride(2),
            BLOCK_F=BLOCK_F
        )

        # Return bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
