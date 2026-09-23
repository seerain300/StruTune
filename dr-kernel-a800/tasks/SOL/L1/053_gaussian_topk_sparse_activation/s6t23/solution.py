import torch
import triton
import triton.language as tl


# Triton kernels

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
    tl.store(out_sum_ptr + pid, total_sum)


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
    tl.store(out_sumsq_ptr + pid, total_sumsq)


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
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var)

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(out_ptr, p,
                        a1, a2, a3, a4, a5, a6,
                        b1, b2, b3, b4, b5,
                        c1, c2, c3, c4, c5, c6,
                        d1, d2, d3, d4, p_low):
    # Compute inverse normal CDF at p using A&S 7.1.26 approximation; write to out_ptr[0].
    p = tl.float32(p)
    # Piecewise regions
    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # Mid region
    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    poly_b = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / poly_b
    # Upper region
    mask_high = p > (1.0 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select
    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_high, 0.0)

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, mean_ptr, std_ptr, z_buf_ptr, out_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b, stride_s, stride_f,
                           BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_buf_ptr)  # scalar

    threshold = mean + std * z

    row_start = b * stride_b + s * stride_s
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        x_ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold  # threshold is scalar
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + pid * F + off + tl.arange(0, BLOCK_F), y, mask=mask)


# -----------------------
# ModelNew forward
# -----------------------
class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and 3D shape [batch, seq, features]
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        assert inputs.ndim == 3, "inputs must have shape [batch_size, seq_len, intermediate_size]"
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Prepare outputs and buffers
        out_sum = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        # Buffer for z (scalar)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # Strides
        stride_b, stride_s, stride_f = inputs.stride()

        # Choose block size for features
        if F >= 16384:
            BLOCK_F = 1024
        elif F >= 8192:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # Launch reduction kernels: one program per row
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, out_sum, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)
        sumsq_rows_kernel[grid](inputs, out_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Compute mean and std per row
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std, B, S, F)

        # Compute inverse-normal CDF for target_sparsity using A&S 7.1.26
        # Constants (Abramowitz & Stegun 7.1.26)
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
        ndtri_scalar_kernel[(1,)](z_buf, self.target_sparsity,
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4, p_low)

        # Apply threshold per row: y = max(input - (mean + std * z), 0)
        # Allocate output as float32, then cast to bfloat16 on host
        out = torch.empty((B, S, F), dtype=torch.float32, device=device)
        apply_threshold_kernel[grid](inputs, out_mean, out_std, z_buf, out,
                                     B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Match original: return bfloat16
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
