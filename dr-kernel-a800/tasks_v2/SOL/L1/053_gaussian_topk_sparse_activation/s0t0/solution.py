import torch
import triton
import triton.language as tl


@triton.jit
def mean_lastdim_kernel(
    inputs_ptr,      # *float32, shape [B, S, F]
    mean_out_ptr,    # *float32, shape [B, S, 1] (we index as linear [B*S, 1])
    B: tl.constexpr,
    S: tl.constexpr,
    F,               # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = (b * S + s) * F
    sum_val = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + base + f, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean = sum_val / F
    tl.store(mean_out_ptr + (b * S + s), mean)


@triton.jit
def std_lastdim_kernel(
    inputs_ptr,       # *float32, shape [B, S, F]
    mean_ptr,         # *float32, shape [B, S, 1] (indexed as [B*S, 1])
    std_out_ptr,      # *float32, shape [B, S, 1]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                # int32
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + (b * S + s))
    base = (b * S + s) * F
    sum_sq = 0.0
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + base + f, mask=mask, other=0.0)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)

    var = sum_sq / F
    std = tl.sqrt(var)
    tl.store(std_out_ptr + (b * S + s), std)


@triton.jit
def apply_cutoff_relu_kernel(
    inputs_ptr,       # *float32, shape [B, S, F]
    mean_ptr,         # *float32, shape [B, S, 1]
    std_ptr,          # *float32, shape [B, S, 1]
    output_ptr,       # *float32, shape [B, S, F]
    B: tl.constexpr,
    S: tl.constexpr,
    F,                # int32
    z,                # float32 scalar
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    mean = tl.load(mean_ptr + (b * S + s))
    std = tl.load(std_ptr + (b * S + s))
    cutoff = mean + std * z

    base = (b * S + s) * F
    for off in range(0, F, BLOCK_F):
        f = off + tl.arange(0, BLOCK_F)
        mask = f < F
        x = tl.load(inputs_ptr + base + f, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(output_ptr + base + f, y, mask=mask)


# Optional: a local ndtri function (host-side). Since it's scalar and not
# differentiable, computing it on host is fine. We only use it to compute z.
# The evaluation harness passes target_sparsity as a float to ModelNew.forward.
# Keeping the function available if needed elsewhere.
def _ndtri_approx(p: float) -> float:
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

    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return inputs

        assert inputs.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        original_dtype = inputs.dtype

        # Compute in float32 for stability (reference code does this)
        inputs_f32 = inputs.contiguous().to(torch.float32)
        B, S, F = inputs_f32.shape

        # Allocate mean and std outputs [B, S, 1] in float32
        mean_out = torch.empty((B, S, 1), dtype=torch.float32, device=inputs.device)
        std_out = torch.empty((B, S, 1), dtype=torch.float32, device=inputs.device)

        # Launch mean kernel: one program per row (b, s)
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

        # Compute z = inverse normal CDF of target_sparsity (scalar)
        z = float(_ndtri_approx(target_sparsity))

        # Allocate output buffer
        output_f32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs.device)

        # Launch apply kernel
        apply_cutoff_relu_kernel[(B * S,)](
            inputs_f32, mean_out, std_out, output_f32,
            B=B, S=S, F=F, z=z, BLOCK_F=BLOCK_F,
        )

        # Cast back to original dtype
        return output_f32.to(original_dtype)


def run(*args):
    return ModelNew()(*args)
