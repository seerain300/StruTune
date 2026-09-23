import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _ndtri(p: float) -> float:
    """
    Inverse of the standard normal CDF (quantile function) using Abramowitz & Stegun 7.1.26 approximation.
    Returns a Python float (scalar).
    """
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Constants for approximation
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

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = c1; den = d1
        for k in range(1, 6):
            num = num * q + globals()[f'c{k}']
        for k in range(1, 4):
            den = den * q + globals()[f'd{k}']
        z = num / (den * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = a1; den = b1
        for k in range(1, 6):
            num = num * r + globals()[f'a{k}']
        for k in range(1, 5):
            den = den * r + globals()[f'b{k}']
        z = (num * q) / (den * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = c1; den = d1
        for k in range(1, 6):
            num = num * q + globals()[f'c{k}']
        for k in range(1, 4):
            den = den * q + globals()[f'd{k}']
        z = -num / (den * q + 1.0)
    return z


@triton.jit
def sum_rows_kernel(inputs_ptr, out_sum_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)  # row id in [0, B*S)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    row_sum = 0.0
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        row_sum += tl.sum(x, axis=0)
        offs += BLOCK_F
    tl.atomic_add(out_sum_ptr + pid, row_sum)


@triton.jit
def sumsq_rows_kernel(inputs_ptr, out_sumsq_ptr, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)  # row id in [0, B*S)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    row_sumsq = 0.0
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        row_sumsq += tl.sum(x * x, axis=0)
        offs += BLOCK_F
    tl.atomic_add(out_sumsq_ptr + pid, row_sumsq)


@triton.jit
def apply_threshold_kernel(inputs_ptr, out_ptr, mean_ptr, std_ptr, z_val, B, S, F, stride_b, stride_s, stride_f, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)  # row id in [0, B*S)
    b = pid // S
    s = pid % S
    base = b * stride_b + s * stride_s

    threshold = mean_ptr[pid] + std_ptr[pid] * z_val  # z_val is a scalar float
    offs = 0
    while offs < F:
        idx = offs + tl.arange(0, BLOCK_F)
        mask = idx < F
        ptrs_in = inputs_ptr + base + idx * stride_f
        x = tl.load(ptrs_in, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x - threshold, 0.0)
        ptrs_out = out_ptr + base + idx * stride_f
        tl.store(ptrs_out, y, mask=mask)
        offs += BLOCK_F


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of run:
        - Computes per-(batch, seq) mean and std over feature dim (population std, unbiased=False).
        - Uses inverse-normal CDF (Abramowitz & Stegun) for z = ndtri(target_sparsity).
        - Applies y = max(input - (mean + std*z), 0) and returns bfloat16.
        """
        # Fallback to torch if not CUDA
        if not inputs.is_cuda:
            inputs_f32 = inputs.to(torch.float32)
            mean = inputs_f32.mean(dim=-1, keepdim=True)
            std = inputs_f32.std(dim=-1, keepdim=True, unbiased=False)
            z = _ndtri(target_sparsity)
            threshold = mean + std * z
            y = F.relu(inputs_f32 - threshold)
            return y.to(torch.bfloat16)

        # Ensure contiguous
        inputs = inputs.contiguous()
        B, S, F = inputs.shape
        device = inputs.device

        # Output buffer (float32 for kernel writes)
        out = torch.empty_like(inputs, dtype=torch.float32, device=device)

        # Strides
        stride_b, stride_s, stride_f = inputs.stride()

        # Choose BLOCK_F based on F
        BLOCK_F = 1024 if F >= 1024 else (512 if F >= 512 else 256)

        # Allocate per-row accumulators
        row_sum = torch.zeros((B * S,), dtype=torch.float32, device=device)
        row_sumsq = torch.zeros((B * S,), dtype=torch.float32, device=device)

        # Launch reduction kernels
        grid = (B * S,)
        sum_rows_kernel[grid](inputs, row_sum, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)
        sumsq_rows_kernel[grid](inputs, row_sumsq, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Compute mean and std (population)
        mean = row_sum / F
        var = row_sumsq / F - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # Compute z using host-side _ndtri; tiny numerical difference is acceptable.
        z_val = float(_ndtri(target_sparsity))

        # Launch apply kernel
        apply_threshold_kernel[grid](inputs, out, mean, std, z_val, B, S, F, stride_b, stride_s, stride_f, BLOCK_F=BLOCK_F)

        # Cast to bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
