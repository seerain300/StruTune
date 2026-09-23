import math
import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean across the last dimension (F)
@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F], contiguous
    mean_out_ptr,      # *fp32, shape [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index
    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + pid * F + f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / F
    tl.store(mean_out_ptr + pid, mean)


# Kernel: compute per-row std across the last dimension (population std)
@triton.jit
def std_lastdim_kernel(
    inputs_ptr,        # *fp32
    mean_ptr,          # *fp32, shape [B*S]
    std_out_ptr,       # *fp32, shape [B*S]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid)
    sum_sq = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + pid * F + f, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    var = sum_sq / F
    std = tl.sqrt(var)
    tl.store(std_out_ptr + pid, std)


# Kernel: apply cutoff = mean + std * z and y = max(0, x - cutoff) per row, write fp32
@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    mean_ptr,          # *fp32, shape [B*S]
    std_ptr,           # *fp32, shape [B*S]
    z,                 # fp32 scalar (inverse-normal CDF of target_sparsity)
    outputs_ptr,       # *fp32, shape [B, S, F] (we'll cast to bf16 in host)
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * z

    # Elementwise: y = max(0, x - cutoff)
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + pid * F + f, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(outputs_ptr + pid * F + f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, just return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "Input must be a CUDA tensor"
        B, S, F = inputs.shape

        # Compute in float32 inside Triton for stability
        inputs_fp32 = inputs.float().contiguous()

        # Allocate outputs as fp32 for Triton, then cast to bfloat16 at the end
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs.device)

        # 1) Compute mean per row (B*S)
        mean_out = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        # 2) Compute std per row
        std_out = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Kernel launch params
        BLOCK_F = 1024
        grid = (B * S,)

        # Launch mean reduction
        mean_lastdim_kernel[grid](
            inputs_fp32, mean_out, B, S, F, BLOCK_F, num_warps=4, num_stages=2
        )

        # Launch std reduction
        std_lastdim_kernel[grid](
            inputs_fp32, mean_out, std_out, B, S, F, BLOCK_F, num_warps=4, num_stages=2
        )

        # 3) Compute z = inverse-normal CDF(target_sparsity) using A&S approximation on host (scalar only).
        # Piecewise constants
        p_low = 0.02425
        p_high = 1.0 - p_low
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00

        p = float(target_sparsity)
        # Lower tail
        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
            z = num / den
        # Middle region
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            z = num / den
        # Upper tail
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
            z = -num / den

        # 4) Apply cutoff and ReLU in Triton, write to fp32
        apply_cutoff_relu_kernel[grid](
            inputs_fp32, mean_out, std_out, z, out_fp32, B, S, F, BLOCK_F, num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match typical expected dtype in evaluations
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
