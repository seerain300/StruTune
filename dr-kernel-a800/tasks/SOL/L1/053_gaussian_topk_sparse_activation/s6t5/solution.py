import torch
import triton
import triton.language as tl


@triton.jit
def single_reduce_rows_kernel(
    inp_ptr,                 # *float32
    out_sum_ptr,             # *float32 (shape [B*S])
    out_sumsq_ptr,           # *float32 (shape [B*S])
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    # One program per row (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Base pointer for this row
    base = inp_ptr + b * stride_b + s * stride_s

    total_sum = 0.0
    total_sumsq = 0.0

    # Loop over feature dimension in chunks
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        ptrs = base + offs * stride_f
        vals = tl.load(ptrs, mask=mask, other=0.0)

        # Reduce vector to scalars
        partial_sum = tl.sum(vals, axis=0)
        partial_sumsq = tl.sum(vals * vals, axis=0)

        total_sum += partial_sum
        total_sumsq += partial_sumsq

    # Atomic add to per-row accumulators (one per program)
    tl.atomic_add(out_sum_ptr + pid, total_sum)
    tl.atomic_add(out_sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_stats_kernel(
    out_sum_ptr,     # *float32 [B*S]
    out_sumsq_ptr,   # *float32 [B*S]
    out_mean_ptr,    # *float32 [B*S]
    out_std_ptr,     # *float32 [B*S]
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
):
    pid = tl.program_id(0)
    total_sum = tl.load(out_sum_ptr + pid)
    total_sumsq = tl.load(out_sumsq_ptr + pid)

    mean = total_sum / F
    var = total_sumsq / F - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to rounding
    std = tl.sqrt(var)

    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    z_out_ptr,        # *float32, length 1
    p,                # float32 scalar (target_sparsity)
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,
):
    # Compute inverse-normal CDF via Abramowitz & Stegun 7.1.26 approximation.
    # p is scalar in (0,1). We will implement piecewise regions via masks and vector ops.
    p_buf = tl.full((1,), p, tl.float32)
    x_buf = tl.full((1,), 0.0, tl.float32)  # dummy to make Triton happy; not used

    # Lower region
    mask_low = p_buf > 0.0
    # Note: Triton doesn't allow branching on tensors using Python ifs; use masks.
    # We'll compute the result vector and then take the [0] element.
    q_low = tl.sqrt(-2.0 * tl.log(p_buf))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    res_low = poly_low / denom_low

    # Mid region
    mask_mid = (p_buf >= 0.0) & (p_buf <= 1.0)
    q_mid = p_buf - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    res_mid = poly_mid * q_mid / denom_mid

    # Upper region
    mask_high = p_buf < 1.0
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p_buf))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    res_high = -poly_high / denom_high

    # Combine results using masks; but since p is scalar, we can select the appropriate result.
    # Triton will evaluate all branches; we just need the final scalar. We'll use res_mid as default.
    res = res_mid
    # Replace with lower/higher if mask matches. For scalars, this is fine.
    res = tl.where(mask_low, res_low, res)
    res = tl.where(mask_high, res_high, res)

    # Store result
    tl.store(z_out_ptr, res)


@triton.jit
def apply_threshold_kernel(
    inp_ptr,          # *float32 [B, S, F]
    mean_ptr,         # *float32 [B*S]
    std_ptr,          # *float32 [B*S]
    z_ptr,            # *float32 [1]
    out_ptr,          # *float32 [B, S, F] (we'll cast to bfloat16 on host after)
    B: tl.constexpr, S: tl.constexpr, F: tl.constexpr,
    stride_b, stride_s, stride_f,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    base = inp_ptr + b * stride_b + s * stride_s
    out_base = out_ptr + b * stride_b + s * stride_s

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)

    threshold = mean + std * z

    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        in_ptrs = base + offs * stride_f
        out_ptrs = out_base + offs * stride_f

        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        y = vals - threshold  # broadcast threshold scalar
        y = tl.maximum(y, 0.0)
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure inputs are CUDA tensors
        assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels"
        device = inputs.device
        # Keep original dtype for reading, compute in float32
        input_bf16 = inputs
        # Convert to float32 for numerics; keep it on device
        input_f32 = inputs.to(torch.float32)

        B, S, F = input_f32.shape
        stride_b, stride_s, stride_f = input_f32.stride()

        # Choose a BLOCK_F heuristic: power of two up to 1024
        if F >= 16384:
            BLOCK_F = 1024
        elif F >= 8192:
            BLOCK_F = 1024
        elif F >= 4096:
            BLOCK_F = 512
        else:
            BLOCK_F = 256

        # Allocate per-row sums
        out_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        out_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch single-pass reduction kernel
        grid = (B * S,)
        single_reduce_rows_kernel[grid](
            input_f32,
            out_sum,
            out_sumsq,
            B, S, F,
            stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
        )

        # Compute mean and std per row
        out_mean = torch.empty((B * S,), dtype=torch.float32, device=device)
        out_std = torch.empty((B * S,), dtype=torch.float32, device=device)
        compute_stats_kernel[(B * S,)](
            out_sum,
            out_sumsq,
            out_mean,
            out_std,
            B, S, F,
        )

        # Compute inverse-normal CDF for scalar target_sparsity
        z_buf = torch.empty((1,), dtype=torch.float32, device=device)
        # Constants for A&S 7.1.26
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

        ndtri_scalar_kernel[(1,)](
            z_buf,
            target_sparsity,  # pass scalar as float
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low,
        )

        # Apply threshold and write to output (float32)
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=device)
        apply_threshold_kernel[grid](
            input_f32,
            out_mean,
            out_std,
            z_buf,
            out_f32,
            B, S, F,
            stride_b, stride_s, stride_f,
            BLOCK_F=BLOCK_F,
        )

        # Cast to bfloat16 to match original Model's return type
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
