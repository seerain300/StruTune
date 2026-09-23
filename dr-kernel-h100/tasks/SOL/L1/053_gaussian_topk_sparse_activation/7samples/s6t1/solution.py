import math
import torch
import torch.nn.functional as F

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def _rowwise_mean_std_kernel(
    X_ptr,          # *fp32
    B, S, F,        # int32
    means_ptr,      # *fp32, shape [B*S]
    stds_ptr,       # *fp32, shape [B*S]
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Map program id to (b, s)
    b = pid // S
    s = pid % S
    # Compute row base pointer for (b, s, :)
    row_start = (b * S + s) * F
    # Accumulators in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over last dimension in blocks
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        # Load a block of the row; cast to fp32 for stability
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and std (population, unbiased=False)
    n = F
    mean = sum_val / n
    # std = sqrt(sum_sq / n - mean^2)
    var = sum_sq / n - mean * mean
    # Avoid negative due to numerical error: clamp to >= 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store per-row results
    tl.store(means_ptr + pid, mean)
    tl.store(stds_ptr + pid, std)


@triton.jit
def _sparsify_relu_kernel(
    X_ptr,            # *fp32 input
    B, S, F,          # int32
    means_ptr,        # *fp32, shape [B*S]
    stds_ptr,         # *fp32, shape [B*S]
    std_multiplier,   # fp32 scalar
    OUT_ptr,          # *fp32 output
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Map program id to (b, s)
    b = pid // S
    s = pid % S

    # Load per-row mean and std
    mean = tl.load(means_ptr + pid)
    std = tl.load(stds_ptr + pid)
    # Compute threshold = mean + std * multiplier (broadcast across F)
    threshold = mean + std * std_multiplier

    row_start = (b * S + s) * F
    # Elementwise sparsification: OUT = max(0, X - threshold)
    for offs in range(0, F, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < F
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        # ReLU: max(0, y)
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + row_start + idx, y, mask=mask)


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun 5.2.23 approximation.
    Input: p in (0, 1)
    Output: z such that P(Z <= z) = p for standard normal Z.
    """
    # Constants
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
        q = math.sqrt(-2.0 * math.log(p))
        res = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return res
    elif p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        res = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return res
    else:
        q = p - 0.5
        r = q * q
        res = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
              (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return res


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        """
        Triton-optimized forward:
        - Compute per-row mean and std along last dim using Triton.
        - Compute threshold = mean + std * inv_norm_cdf(target_sparsity).
        - Sparsify via ReLU(input - threshold) using Triton.
        Returns tensor of shape [batch_size, seq_len, intermediate_size] in bfloat16.
        """
        # Handle empty tensors
        if inputs.numel() == 0:
            return inputs.new_empty(inputs.shape, dtype=torch.bfloat16)

        # Ensure CUDA tensor for Triton
        assert inputs.is_cuda, "ModelNew.forward requires a CUDA tensor for Triton execution."
        # We will compute in fp32 inside Triton, and return bfloat16
        B, S, F = inputs.shape

        # Create contiguous fp32 copy for Triton kernels
        x_fp32 = inputs.to(torch.float32).contiguous()

        # Allocate buffers for per-row mean and std
        means = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)
        stds = torch.empty((B * S,), dtype=torch.float32, device=inputs.device)

        # Launch rowwise mean & std kernel: one program per (b, s)
        BLOCK = 1024
        grid = (B * S,)
        _rowwise_mean_std_kernel[grid](
            x_fp32, B, S, F,
            means, stds,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute std_multiplier as scalar (host-side, no torch ops)
        std_multiplier = float(_ndtri(float(target_sparsity)))

        # Allocate output (fp32) for sparsification
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs.device)

        # Launch sparsification kernel: one program per (b, s)
        _sparsify_relu_kernel[grid](
            x_fp32, B, S, F,
            means, stds,
            std_multiplier,
            out_fp32,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
