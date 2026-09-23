import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    x_ptr,                 # *const float (input)
    out_sum_ptr,           # *float32, shape [B*S]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # program id: 0..(B*S-1)
    b = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_F)
    total = 0.0
    for f in range(0, F, BLOCK_F):
        idx = f + offs
        mask = idx < F
        x = tl.load(x_ptr + b * stride_b + s * stride_s + idx * stride_f, mask=mask, other=0.0)
        x = x.to(tl.float32)
        total += tl.sum(x, axis=0)
    tl.atomic_add(out_sum_ptr + pid, total)


@triton.jit
def sumsq_rows_kernel(
    x_ptr,                 # *const float (input)
    out_sumsq_ptr,         # *float32, shape [B*S]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    pid = tl.program_id(0)  # program id: 0..(B*S-1)
    b = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_F)
    total = 0.0
    for f in range(0, F, BLOCK_F):
        idx = f + offs
        mask = idx < F
        x = tl.load(x_ptr + b * stride_b + s * stride_s + idx * stride_f, mask=mask, other=0.0)
        x = x.to(tl.float32)
        total += tl.sum(x * x, axis=0)
    tl.atomic_add(out_sumsq_ptr + pid, total)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,        # *float32, shape [B*S]
    out_sumsq_ptr,      # *float32, shape [B*S]
    out_mean_ptr,       # *float32, shape [B*S]
    out_std_ptr,        # *float32, shape [B*S]
    B: tl.constexpr,    # int
    S: tl.constexpr,    # int
    F: tl.constexpr,    # int
):
    pid = tl.program_id(0)  # program id: 0..(B*S-1)
    sum = tl.load(out_sum_ptr + pid)
    sumsq = tl.load(out_sumsq_ptr + pid)
    # mean and std along feature dimension (population std)
    mean = sum / F
    var = sumsq / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def compute_z_kernel(
    out_z_ptr,          # *float32, shape [1] (scalar output)
    p,                  # float32 scalar (target_sparsity)
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,
):
    # Abramowitz & Stegun 7.1.26 approximation for ndtri(p)
    # piecewise regions
    mask_low = p < p_low
    mask_high = p > (1.0 - p_low)
    mask_mid = ~(mask_low | mask_high)

    # lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # mid region
    q_mid = p - 0.5
    r = q_mid * q_mid
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q_mid / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # select: compute z_mid, then overwrite via arithmetic branching
    z = z_mid
    z = z + (z_low - z) * mask_low + (z_high - z) * mask_high
    tl.store(out_z_ptr, z)


@triton.jit
def apply_threshold_kernel(
    x_ptr,               # *const float (input, will be cast to float32 inside)
    mean_ptr,            # *const float32, shape [B*S]
    std_ptr,             # *const float32, shape [B*S]
    out_ptr,             # *float32, shape [B,S,F] (output)
    B: tl.constexpr,     # int
    S: tl.constexpr,     # int
    F: tl.constexpr,     # int
    z,                   # float32 scalar (from device)
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    out_stride_b: tl.constexpr,  # int
    out_stride_s: tl.constexpr,  # int
    out_stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,       # int
):
    pid = tl.program_id(0)  # program id: 0..(B*S-1)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * z

    offs = tl.arange(0, BLOCK_F)
    for f in range(0, F, BLOCK_F):
        idx = f + offs
        mask = idx < F
        x = tl.load(x_ptr + b * stride_b + s * stride_s + idx * stride_f, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - threshold  # threshold is scalar
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + b * out_stride_b + s * out_stride_s + idx * out_stride_f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, F], dtype bfloat16 (or float), device CUDA
        assert x.is_cuda, "ModelNew requires CUDA tensor input for Triton kernels."
        assert x.ndim == 3, "Input must be 3D [batch_size, seq_len, intermediate_size]."
        B, S, F = x.shape
        device = x.device

        # Make input contiguous for predictable strides
        x = x.contiguous()

        # Output buffer in float32 for computation
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        # Prepare per-row accumulators (sum and sum of squares)
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)

        # Choose block size along feature dimension
        BLOCK_F = 1024 if F >= 1024 else (512 if F >= 512 else 256)

        # 1) Compute sum and sum of squares per row
        grid = (B * S,)
        sum_rows_kernel[grid](
            x,
            out_sum,
            B,
            S,
            F,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            BLOCK_F=BLOCK_F,
        )
        sumsq_rows_kernel[grid](
            x,
            out_sumsq,
            B,
            S,
            F,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # 2) Compute mean and std per row
        compute_stats_kernel[grid](
            out_sum,
            out_sumsq,
            out_mean,
            out_std,
            B,
            S,
            F,
        )

        # 3) Compute inverse-normal CDF z in Triton (scalar)
        out_z = torch.empty((1,), dtype=torch.float32, device=device)
        # Constants for A&S 7.1.26
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_z_kernel[(1,)](
            out_z,
            self.target_sparsity,  # scalar float
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
        )

        # Read scalar z
        z = out_z[0].item()

        # 4) Apply threshold in Triton
        apply_threshold_kernel[grid](
            x,
            out_mean,
            out_std,
            out_f32,
            B,
            S,
            F,
            float(z),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out_f32.stride(0),
            out_f32.stride(1),
            out_f32.stride(2),
            BLOCK_F=BLOCK_F,
        )

        # Cast to bfloat16 to match original return dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
