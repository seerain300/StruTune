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
def compute_stats_kernel(
    out_sum_ptr,          # *const float32 (length B*S)
    out_sumsq_ptr,        # *const float32 (length B*S)
    mean_out_ptr,         # *float32 (length B*S)
    std_out_ptr,          # *float32 (length B*S)
    B: tl.constexpr,      # int
    S: tl.constexpr,      # int
    F: tl.constexpr,      # int
):
    # One program per row pid in [0, B*S)
    pid = tl.program_id(axis=0)
    sum_val = tl.load(out_sum_ptr + pid)
    sumsq_val = tl.load(out_sumsq_ptr + pid)
    # mean and std (population std, unbiased=False)
    mean = sum_val / F
    var = sumsq_val / F - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_out_ptr + pid, mean)
    tl.store(std_out_ptr + pid, std)


@triton.jit
def ndtri_scalar_kernel(
    p_ptr,        # *const float32 (length 1)
    out_ptr,      # *float32 (length 1)
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    p_low: tl.constexpr,
):
    # Single program: compute inverse normal CDF for p[0]
    p = tl.load(p_ptr)  # scalar
    # lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly = c1 * q_low + c2
    poly2 = poly * q_low + c3
    poly3 = poly2 * q_low + c4
    poly4 = poly3 * q_low + c5
    poly5 = poly4 * q_low + c6
    denom = d1 * q_low + d2
    denom2 = denom * q_low + d3
    denom3 = denom2 * q_low + d4
    z_low = -poly5 / denom3

    # central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly_c = a1 * r + a2
    poly2_c = poly_c * r + a3
    poly3_c = poly2_c * r + a4
    poly4_c = poly3_c * r + a5
    poly5_c = poly4_c * r + a6
    denom_c = b1 * r + b2
    denom2_c = denom_c * r + b3
    denom3_c = denom2_c * r + b4
    denom4_c = denom3_c * r + b5
    z_mid = poly5_c * q_mid / denom4_c

    # select region
    if p < p_low:
        z = z_low
    elif p <= (1.0 - p_low):
        z = z_mid
    else:
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly_up = c1 * q_up + c2
        poly2_up = poly_up * q_up + c3
        poly3_up = poly2_up * q_up + c4
        poly4_up = poly3_up * q_up + c5
        poly5_up = poly4_up * q_up + c6
        denom_up = d1 * q_up + d2
        denom2_up = denom_up * q_up + d3
        denom3_up = denom2_up * q_up + d4
        z = -poly5_up / denom3_up

    tl.store(out_ptr, z)


@triton.jit
def apply_threshold_kernel(
    inp_ptr,        # *const float32
    mean_ptr,       # *const float32 (length B*S)
    std_ptr,        # *const float32 (length B*S)
    out_ptr,        # *bfloat16
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

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * tl.load(out_ptr + 0)  # scalar z loaded once

    f_start = 0
    offs = tl.arange(0, BLOCK_F)
    while f_start < F:
        f = f_start + offs
        mask = f < F
        inp_idx = b * stride_b + s * stride_s + f * stride_f
        inp_vals = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
        y = inp_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        out_idx = b * stride_b + s * stride_s + f * stride_f
        # Store as bfloat16
        tl.store(out_ptr + out_idx, y.to(tl.bfloat16), mask=mask)
        f_start += BLOCK_F


class ModelNew(torch.nn.Module):
    def forward(self, input: torch.Tensor, target_sparsity: float):
        """
        Triton-only implementation:
        - Compute per-(b,s) mean and std along last dim (population std, unbiased=False).
        - Compute inverse-normal CDF for target_sparsity via A&S 7.1.26 in Triton.
        - Apply thresholding and ReLU per element in Triton, output as bfloat16.
        """
        if target_sparsity == 0.0:
            # No sparsity, return input as bfloat16
            return input.to(torch.bfloat16)

        # Compute in float32 for stability
        input_f32 = input.to(torch.float32)
        B, S, F = input_f32.shape
        device = input_f32.device

        stride_b, stride_s, stride_f = input_f32.stride()

        # Buffers for sums and sumsq (length B*S)
        out_sum = torch.zeros(B * S, dtype=torch.float32, device=device)
        out_sumsq = torch.zeros(B * S, dtype=torch.float32, device=device)

        mean_out = torch.empty(B * S, dtype=torch.float32, device=device)
        std_out = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_F = 1024
        grid = (B * S,)

        # Reduction kernels: sum and sumsq
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

        # Compute mean and std per row (stats kernel)
        triton.run(compute_stats_kernel, grid=(B * S,),
                   out_sum_ptr=out_sum,
                   out_sumsq_ptr=out_sumsq,
                   mean_out_ptr=mean_out,
                   std_out_ptr=std_out,
                   B=B, S=S, F=F)

        # Compute inverse-normal CDF for target_sparsity using Triton scalar kernel
        p_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
        z_out = torch.empty(1, dtype=torch.float32, device=device)

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

        triton.run(ndtri_scalar_kernel, grid=(1,),
                   p_ptr=p_tensor,
                   out_ptr=z_out,
                   a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6,
                   b1=b1, b2=b2, b3=b3, b4=b4, b5=b5,
                   c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6,
                   d1=d1, d2=d2, d3=d3, d4=d4,
                   p_low=p_low)

        z = z_out[0]  # scalar float32

        # Allocate output tensor as bfloat16 to match original final cast
        output_bf16 = torch.empty((B, S, F), dtype=torch.bfloat16, device=device)

        # Elementwise apply kernel with per-row mean/std and scalar z
        triton.run(apply_threshold_kernel, grid=grid,
                   inp_ptr=input_f32,
                   mean_ptr=mean_out,
                   std_ptr=std_out,
                   out_ptr=output_bf16,
                   B=B, S=S, F=F,
                   stride_b=stride_b, stride_s=stride_s, stride_f=stride_f,
                   BLOCK_F=BLOCK_F)

        return output_bf16


def run(*args):
    return ModelNew()(*args)
