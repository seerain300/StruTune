import math
import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean across the last dimension (F).
@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F] contiguous along last dim
    mean_out_ptr,      # *fp32, shape [B*S]
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + pid * F + f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / F
    tl.store(mean_out_ptr + pid, mean)


# Kernel: compute per-row std across the last dimension (population std).
@triton.jit
def std_lastdim_kernel(
    inputs_ptr,        # *fp32
    mean_ptr,          # *fp32, shape [B*S]
    std_out_ptr,       # *fp32, shape [B*S]
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


# Kernel: compute inverse-normal CDF (ndtri) for scalar p using A&S approximation.
# Launch with grid=(1,) and write scalar z to a 1-element output tensor (z_out_ptr).
@triton.jit
def ndtri_approx_kernel(
    z_out_ptr,         # *fp32, length 1
    p,                 # fp32 scalar input in (0, 1)
    p_low: tl.constexpr,   # 0.02425
    p_high: tl.constexpr,  # 1 - p_low
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
):
    # Piecewise approximation
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # Lower tail: p < 0.02425
    q_low = tl.sqrt(-2.0 * tl.log(p))
    num_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    z_low = num_low / den_low

    # Middle: 0.02425 <= p <= 0.97575
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = num_mid / den_mid

    # Upper tail: p > 0.97575
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    num_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -num_high / den_high

    # Combine
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(z_out_ptr, z)


# Kernel: apply cutoff = mean + std * z and y = max(0, x - cutoff) per row
@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    mean_ptr,          # *fp32, shape [B*S]
    std_ptr,           # *fp32, shape [B*S]
    z_ptr,             # *fp32, length 1 (scalar z)
    out_ptr,           # *fp32, shape [B, S, F]
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar read once per row program
    cutoff = mean + std * z

    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + pid * F + f, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + pid * F + f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs as-is
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA and contiguous; compute in fp32 for stability
        assert inputs.is_cuda, "Inputs must be on CUDA device"
        B, S, F = inputs.shape
        inputs_fp32 = inputs.contiguous().to(torch.float32)

        # Allocate outputs and per-row buffers
        NROWS = B * S
        mean_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)
        std_out = torch.empty(NROWS, dtype=torch.float32, device=inputs.device)

        # Launch mean kernel
        grid = (NROWS,)
        mean_lastdim_kernel[grid](
            inputs_fp32,
            mean_out,
            F,
            BLOCK_F=1024,
        )

        # Launch std kernel
        std_lastdim_kernel[grid](
            inputs_fp32,
            mean_out,
            std_out,
            F,
            BLOCK_F=1024,
        )

        # Prepare constants for A&S approximation
        p = float(target_sparsity)
        p_low = 0.02425
        p_high = 1.0 - p_low

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

        # Output tensor for z (1 element)
        z_out = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Launch ndtri approximation kernel (grid = 1)
        ndtri_approx_kernel[(1,)](
            z_out,
            p,
            p_low,
            p_high,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
        )

        # Read scalar z (host-side, unavoidable)
        z = float(z_out.item())

        # Allocate output (fp32 for computation; cast after)
        out_fp32 = torch.empty_like(inputs_fp32)

        # Launch apply kernel
        apply_cutoff_relu_kernel[grid](
            inputs_fp32,
            mean_out,
            std_out,
            z_out,  # pass pointer; kernel reads scalar z
            out_fp32,
            F,
            BLOCK_F=1024,
        )

        # Cast back to original dtype (original code returns same dtype as inputs)
        return out_fp32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
