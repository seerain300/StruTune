import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean across the last dimension (F)
@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F] contiguous along last dim
    mean_out_ptr,      # *fp32, shape [B*S, 1] (we'll index as [pid])
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_base = (b * S + s) * F

    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean = sum_val / F
    tl.store(mean_out_ptr + pid, mean)


# Kernel: compute per-row std across the last dimension (population std)
@triton.jit
def std_lastdim_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    mean_ptr,          # *fp32, shape [B*S, 1] (we'll index as [pid])
    std_out_ptr,       # *fp32, shape [B*S, 1]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_base = (b * S + s) * F

    mean = tl.load(mean_ptr + pid)
    sum_sq = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)

    var = sum_sq / F
    std = tl.sqrt(var)
    tl.store(std_out_ptr + pid, std)


# Kernel: compute inverse normal CDF (ndtri) for a scalar p using A&S approximation.
# We will launch this with grid=(1,) and write to a 1-element output tensor.
@triton.jit
def ndtri_approx_kernel(
    out_ptr,           # *fp32, length 1 (output z)
    p,                 # fp32 scalar input
    p_low: tl.constexpr,   # 0.02425
    p_high: tl.constexpr,  # 1 - p_low
    a1, a2, a3, a4, a5, a6,
    b1, b2, b3, b4, b5,
    c1, c2, c3, c4, c5, c6,
    d1, d2, d3, d4,
    BLOCK: tl.constexpr,
):
    # Single program instance, scalar compute
    # Lower tail
    q1 = tl.sqrt(-2.0 * tl.log(p))
    z1 = (((((c1 * q1 + c2) * q1 + c3) * q1 + c4) * q1 + c5) * q1 + c6) / \
         ((((d1 * q1 + d2) * q1 + d3) * q1 + d4) * q1 + 1.0)

    # Middle region
    q2 = p - 0.5
    r2 = q2 * q2
    z2 = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * q2 / \
         (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)

    # Upper tail
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z3 = -(((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6) / \
         ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)

    # Select based on p
    cond_low = p < p_low
    cond_mid = (p >= p_low) & (p <= p_high)

    z = tl.where(cond_low, z1, 0.0)
    z = tl.where(cond_mid, z2, z)
    z = tl.where(~(cond_low | cond_mid), z3, z)  # else branch

    tl.store(out_ptr, z)


# Kernel: elementwise apply cutoff and ReLU
@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,        # *fp32, shape [B, S, F]
    mean_ptr,          # *fp32, shape [B*S, 1]
    std_ptr,           # *fp32, shape [B*S, 1]
    z_ptr,             # *fp32, shape [1], holds scalar z
    output_ptr,        # *fp32, shape [B, S, F]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                 # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_base = (b * S + s) * F

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    # Read scalar z from 1-element tensor
    z = tl.load(z_ptr)
    cutoff = mean + std * z

    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + row_base + f, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(output_ptr + row_base + f, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Early return if no sparsity requested
        if target_sparsity == 0.0:
            return inputs

        # Ensure CUDA tensor
        assert inputs.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        original_dtype = inputs.dtype

        # Compute in float32 for stability
        inputs_f32 = inputs.contiguous().to(torch.float32)
        B, S, F = inputs_f32.shape

        # Allocate per-row mean and std (shape [B*S, 1])
        mean_out = torch.empty((B * S, 1), dtype=torch.float32, device=inputs.device)
        std_out = torch.empty((B * S, 1), dtype=torch.float32, device=inputs.device)

        # Launch mean kernel: one program per row
        BLOCK_F = 256
        mean_lastdim_kernel[(B * S,)](
            inputs_f32, mean_out,
            B=B, S=S, F=F, BLOCK_F=BLOCK_F,
        )

        # Launch std kernel
        std_lastdim_kernel[(B * S,)](
            inputs_f32, mean_out, std_out,
            B=B, S=S, F=F, BLOCK_F=BLOCK_F,
        )

        # Compute z = inverse normal CDF of target_sparsity using Triton, write to 1-element tensor
        z_out = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Constants for A&S approximation
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
        p_high = 1.0 - p_low

        # Launch ndtri approximation kernel with scalar target_sparsity
        ndtri_approx_kernel[(1,)](
            z_out, target_sparsity,
            p_low=p_low, p_high=p_high,
            a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6,
            b1=b1, b2=b2, b3=b3, b4=b4, b5=b5,
            c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6,
            d1=d1, d2=d2, d3=d3, d4=d4,
        )

        # Allocate output buffer
        output_f32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs.device)

        # Launch apply cutoff + ReLU kernel
        apply_cutoff_relu_kernel[(B * S,)](
            inputs_f32, mean_out, std_out, z_out, output_f32,
            B=B, S=S, F=F, BLOCK_F=BLOCK_F,
        )

        # Cast back to original dtype
        return output_f32.to(original_dtype)


def run(*args):
    return ModelNew()(*args)
