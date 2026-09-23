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
    # Compute inverse-normal CDF for p using Abramowitz & Stegun 7.1.26
    # Store result into out_ptr[0]
    p = tl.float32(p)

    # Regions
    low_mask = p < p_low
    mid_mask = (p >= p_low) & (p <= (1.0 - p_low))
    high_mask = p > (1.0 - p_low)

    # Lower region: p < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Mid region: p_low <= p <= 1 - p_low
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly * q_mid / den

    # Upper region: p > 1 - p_low
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select based on masks; combine
    z = tl.where(low_mask, z_low, 0.0) + tl.where(mid_mask, z_mid, 0.0) + tl.where(high_mask, z_high, 0.0)

    # Write to out_ptr[0]
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(X_ptr, out_ptr,
                           mean_ptr, std_ptr, z_ptr,
                           B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
                           stride_b, stride_s, stride_f,
                           BLOCK_F: tl.constexpr):
    # One program per row (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_start = b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar

    threshold = mean + std * z
    # Iterate across F in chunks
    for off in range(0, F, BLOCK_F):
        cols = off + tl.arange(0, BLOCK_F)
        mask = cols < F
        x_ptrs = X_ptr + row_start + cols * stride_f
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        out_ptrs = out_ptr + pid * F + off + tl.arange(0, BLOCK_F)
        tl.store(out_ptrs, y, mask=mask)


# -----------------------
# ModelNew forward
# -----------------------
class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs of shape [batch_size, seq_len, intermediate_size]
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        assert inputs.ndim == 3, "Inputs must be 3D [batch, seq, feature]."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Choose BLOCK_F (chunk size along feature dim)
        # Use a power-of-two up to 1024 for good throughput
        if F >= 1024:
            BLOCK_F = 1024
        elif F >= 512:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # Prepare output and stats tensors
        out_sum = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        # z buffer for inverse-normal CDF (scalar on device)
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)

        # Strides for Triton
        stride_b = inputs.stride(0)
        stride_s = inputs.stride(1)
        stride_f = inputs.stride(2)

        # 1) Compute sum and sumsq per row
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, out_sum,
                              B, S, F,
                              stride_b, stride_s, stride_f,
                              BLOCK_F=BLOCK_F)
        sumsq_rows_kernel[grid](inputs, out_sumsq,
                                B, S, F,
                                stride_b, stride_s, stride_f,
                                BLOCK_F=BLOCK_F)

        # 2) Compute mean and std per row
        compute_stats_kernel[grid](out_sum, out_sumsq, out_mean, out_std,
                                   B, S, F)

        # 3) Compute inverse-normal CDF for target_sparsity (scalar) using Triton
        # A&S constants
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        ndtri_scalar_kernel[(1,)](z_buf, self.target_sparsity,
                                  a1, a2, a3, a4, a5, a6,
                                  b1, b2, b3, b4, b5,
                                  c1, c2, c3, c4, c5, c6,
                                  d1, d2, d3, d4, p_low)

        # 4) Apply threshold and ReLU in Triton: output is float32
        out = torch.empty((B * S * F,), dtype=torch.float32, device=device)
        apply_threshold_kernel[(B * S,)](
            inputs, out,
            out_mean, out_std, z_buf,
            B, S, F,
            stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F
        )

        # 5) Cast to bfloat16 to match original behavior
        return out.view(B, S, F).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
