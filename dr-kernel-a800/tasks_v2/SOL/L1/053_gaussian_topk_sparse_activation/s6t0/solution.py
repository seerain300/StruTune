import torch
import triton
import triton.language as tl


@triton.jit
def sum_rows_kernel(
    x_ptr,           # *const float32
    out_ptr,         # *float32 (length B*S)
    B: tl.constexpr, # int
    S: tl.constexpr, # int
    F: tl.constexpr, # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    total_sum = 0.0
    f_start = 0
    offs = tl.arange(0, BLOCK_F)
    while f_start < F:
        f = f_start + offs
        mask = f < F
        x_idx = b * stride_b + s * stride_s + f * stride_f
        vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        total_sum += tl.sum(vals, axis=0)
        f_start += BLOCK_F

    tl.atomic_add(out_ptr + pid, total_sum)


@triton.jit
def sumsq_rows_kernel(
    x_ptr,           # *const float32
    out_ptr,         # *float32 (length B*S)
    B: tl.constexpr, # int
    S: tl.constexpr, # int
    F: tl.constexpr, # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    total_sumsq = 0.0
    f_start = 0
    offs = tl.arange(0, BLOCK_F)
    while f_start < F:
        f = f_start + offs
        mask = f < F
        x_idx = b * stride_b + s * stride_s + f * stride_f
        vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        total_sumsq += tl.sum(vals * vals, axis=0)
        f_start += BLOCK_F

    tl.atomic_add(out_ptr + pid, total_sumsq)


@triton.jit
def ndtri_kernel(
    p_ptr,           # *const float32, scalar tensor of shape [1]
    out_ptr,         # *float32, scalar tensor of shape [1]
    # A&S coefficients for normal inverse CDF approximation
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low,
):
    # Load scalar p
    p = tl.load(p_ptr)  # float32 scalar
    # Lower region
    mask_low = p < p_low
    # Initialize z to 0
    z = 0.0
    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly1 = c1 * q + c2
        poly2 = poly1 * q + c3
        poly3 = poly2 * q + c4
        poly4 = poly3 * q + c5
        poly5 = poly4 * q + c6
        denom = d1 * q + d2
        denom2 = denom * q + d3
        denom3 = denom2 * q + d4
        z = poly5 / denom3
    else:
        # Central region
        mask_mid = (p >= p_low) & (p <= 1.0 - p_low)
        if mask_mid:
            q = p - 0.5
            r = q * q
            poly6 = a1 * r + a2
            poly7 = poly6 * r + a3
            poly8 = poly7 * r + a4
            poly9 = poly8 * r + a5
            poly10 = poly9 * r + a6
            denom4 = b1 * r + b2
            denom5 = denom4 * r + b3
            denom6 = denom5 * r + b4
            denom7 = denom6 * r + b5
            z = poly10 * q / denom7
        else:
            # Upper region
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly1 = c1 * q + c2
            poly2 = poly1 * q + c3
            poly3 = poly2 * q + c4
            poly4 = poly3 * q + c5
            poly5 = poly4 * q + c6
            denom = d1 * q + d2
            denom2 = denom * q + d3
            denom3 = denom2 * q + d4
            z = -poly5 / denom3
    # Store result
    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(
    inp_ptr,        # *const float32
    out_ptr,        # *float32
    thr_ptr,        # *const float32 (length B*S)
    B: tl.constexpr, # int
    S: tl.constexpr, # int
    F: tl.constexpr, # int
    stride_b: tl.constexpr,  # int
    stride_s: tl.constexpr,  # int
    stride_f: tl.constexpr,  # int
    BLOCK_F: tl.constexpr,   # int
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    threshold = tl.load(thr_ptr + pid)

    f_start = 0
    offs = tl.arange(0, BLOCK_F)
    while f_start < F:
        f = f_start + offs
        mask = f < F
        inp_idx = b * stride_b + s * stride_s + f * stride_f
        inp_vals = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
        y = tl.maximum(inp_vals - threshold, 0.0)
        out_idx = b * stride_b + s * stride_s + f * stride_f
        tl.store(out_ptr + out_idx, y, mask=mask)
        f_start += BLOCK_F


class ModelNew(torch.nn.Module):
    def forward(self, input: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized version of the original run.
        Computes per-(b,s) mean and std along last dim, forms threshold = mean + std * _ndtri(target_sparsity),
        then applies ReLU(input - threshold). Returns bfloat16, matching the original behavior.
        """
        if target_sparsity == 0.0:
            return input.to(torch.bfloat16)

        # Compute in float32 for stability, cast at end
        input_f32 = input.to(torch.float32)
        B, S, F = input_f32.shape
        device = input_f32.device

        stride_b, stride_s, stride_f = input_f32.stride()

        # Buffers for sums and sumsq (length B*S)
        out_sum = torch.zeros(B * S, dtype=torch.float32, device=device)
        out_sumsq = torch.zeros(B * S, dtype=torch.float32, device=device)

        BLOCK_F = 1024
        grid = (B * S,)

        # Reduction kernels
        triton.run(sum_rows_kernel, grid=grid,
                   x_ptr=input_f32,
                   out_ptr=out_sum,
                   B=B, S=S, F=F,
                   stride_b=stride_b, stride_s=stride_s, stride_f=stride_f,
                   BLOCK_F=BLOCK_F)

        triton.run(sumsq_rows_kernel, grid=grid,
                   x_ptr=input_f32,
                   out_ptr=out_sumsq,
                   B=B, S=S, F=F,
                   stride_b=stride_b, stride_s=stride_s, stride_f=stride_f,
                   BLOCK_F=BLOCK_F)

        # Compute mean and std per row
        mean = out_sum / F
        var = out_sumsq / F - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # Constants for A&S approximation (same as the original _ndtri)
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

        # Allocate scalar output for inverse normal CDF
        z_out = torch.empty(1, dtype=torch.float32, device=device)

        # Launch Triton _ndtri kernel for scalar p
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
        triton.run(ndtri_kernel, grid=(1,),
                   p_ptr=p_tensor,
                   out_ptr=z_out,
                   a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6,
                   b1=b1, b2=b2, b3=b3, b4=b4, b5=b5,
                   c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6,
                   d1=d1, d2=d2, d3=d3, d4=d4,
                   p_low=p_low)

        z = z_out[0]  # scalar float32

        # threshold per row: mean + std * z
        threshold = mean + std * z

        # Allocate output tensor (float32 during compute)
        output_f32 = torch.empty_like(input_f32)

        # Elementwise apply kernel
        triton.run(apply_threshold_kernel, grid=grid,
                   inp_ptr=input_f32,
                   out_ptr=output_f32,
                   thr_ptr=threshold,
                   B=B, S=S, F=F,
                   stride_b=stride_b, stride_s=stride_s, stride_f=stride_f,
                   BLOCK_F=BLOCK_F)

        # Return in bfloat16 to match original's final cast
        return output_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
