import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    x_ptr,                 # *const float
    out_sum_ptr,           # *float32, shape [B*S]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
):
    pid = tl.program_id(0)  # program id over rows: 0..(B*S-1)
    b = pid // S
    s = pid % S

    total_sum = 0.0
    # Loop over feature dimension
    for f in range(0, F):
        x_val = tl.load(x_ptr + b * stride_b + s * stride_s + f * stride_f)
        total_sum += x_val.to(tl.float32)

    tl.store(out_sum_ptr + pid, total_sum)


@triton.jit
def sumsq_rows_kernel(
    x_ptr,                 # *const float
    out_sumsq_ptr,         # *float32, shape [B*S]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    total_sumsq = 0.0
    for f in range(0, F):
        x_val = tl.load(x_ptr + b * stride_b + s * stride_s + f * stride_f)
        x_f32 = x_val.to(tl.float32)
        total_sumsq += x_f32 * x_f32

    tl.store(out_sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,           # *float32, shape [B*S]
    out_sumsq_ptr,         # *float32, shape [B*S]
    out_mean_ptr,          # *float32, shape [B*S]
    out_std_ptr,           # *float32, shape [B*S]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
):
    pid = tl.program_id(0)  # 0..(B*S-1)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def compute_z_kernel(
    out_z_ptr,              # *float32, shape [1]
    p,                      # float32
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,                  # float32
):
    # Abramowitz & Stegun 7.1.26 approximation for inverse-normal CDF
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    mask_mid = (p >= p_low) & (p <= (1.0 - p_low))
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid * q_mid / den_mid

    mask_high = p > (1.0 - p_low)
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    z = tl.where(mask_low, z_low, 0.0) + tl.where(mask_mid, z_mid, 0.0) + tl.where(mask_high, z_high, 0.0)
    tl.store(out_z_ptr, z)


@triton.jit
def apply_threshold_kernel(
    x_ptr,                 # *const float (input)
    out_mean_ptr,          # *float32, shape [B*S]
    out_std_ptr,           # *float32, shape [B*S]
    out_ptr,               # *float32, shape [B, S, F]
    B: tl.constexpr,       # int
    S: tl.constexpr,       # int
    F: tl.constexpr,       # int
    z,                     # float32 scalar
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_f: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_f: tl.constexpr,
):
    pid = tl.program_id(0)  # 0..(B*S-1)
    b = pid // S
    s = pid % S

    mean = tl.load(out_mean_ptr + pid)
    std = tl.load(out_std_ptr + pid)
    threshold = mean + std * z

    for f in range(0, F):
        x_val = tl.load(x_ptr + b * stride_b + s * stride_s + f * stride_f).to(tl.float32)
        y_val = tl.maximum(x_val - threshold, 0.0)
        tl.store(out_ptr + b * out_stride_b + s * out_stride_s + f * out_stride_f, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.0):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Precompute z once: z = sqrt(2) * erfinv(2*p - 1)
        z = float(torch.special.erfinv(2.0 * self.target_sparsity - 1.0))
        self.register_buffer("z_buf", torch.tensor([z], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Triton-only forward: no torch ops on data
        assert inputs.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Output buffers (float32 for computation)
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_f32 = torch.empty_like(inputs, dtype=torch.float32, device=device)

        grid = (B * S,)

        # 1) Compute sums and sumsq per row (scalar per-row loop)
        sum_rows_kernel[grid](
            inputs,
            out_sum,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
        )
        sumsq_rows_kernel[grid](
            inputs,
            out_sumsq,
            B, S, F,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
        )

        # 2) Compute mean and std per row
        compute_stats_kernel[grid](
            out_sum, out_sumsq, out_mean, out_std, B, S, F,
        )

        # 3) Precompute or pass z (already computed in __init__)
        z = float(self.z_buf.item())

        # 4) Apply threshold and write output
        apply_threshold_kernel[grid](
            inputs,
            out_mean,
            out_std,
            out_f32,
            B, S, F,
            z,
            inputs.stride(0), inputs.stride(1), inputs.stride(2),
            out_f32.stride(0), out_f32.stride(1), out_f32.stride(2),
        )

        # Return bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
